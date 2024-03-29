import os, sys, time
from os.path import exists
import torch
from models.pix2pix_model import Pix2PixModel
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import torch.optim.lr_scheduler as lr_scheduler
import wandb
from utils.data_loader_multifiles import get_data_loader
from utils.weighted_acc_rmse import weighted_acc_torch_channels, unlog_tp_torch
from utils.viz import viz_fields
import numpy as np
import matplotlib.pyplot as plt
import logging
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap as ruamelDict

from utils.data_loader import get_data_loader_distributed
from utils import get_data_loader_distributed, lr_schedule


class Pix2PixTrainer():
    """
    Trainer object creates the model and optimizers, and uses them to
    updates the weights of the network while reporting losses
    and the latest visuals to visualize the progress in training.
    """

    def __init__(self, params, args):

        self.sweep_id = args.sweep_id
        self.root_dir = args.root_dir
        self.config = args.config

        params.amp = args.amp
        self.world_size = 1
        if 'WORLD_SIZE' in os.environ:
            self.world_size = int(os.environ['WORLD_SIZE'])

        self.local_rank = 0
        self.world_rank = 0
        if self.world_size > 1:
            dist.init_process_group(backend='nccl',
                                    init_method='env://')
            self.world_rank = dist.get_rank()
            self.local_rank = int(os.environ["LOCAL_RANK"])

        torch.cuda.set_device(self.local_rank)
        torch.backends.cudnn.benchmark = True
        
        if self.world_rank==0:
            params.log()
        self.log_to_screen = params.log_to_screen and self.world_rank==0
        self.log_to_wandb = params.log_to_wandb and self.world_rank==0
        params.name = args.config

        self.device = torch.cuda.current_device()
        self.params = params


    def build_and_launch(self):
        # init wandb
        if self.log_to_wandb:
            print(f'log_to_wandb -------------------------------------------------------------------')
            if self.sweep_id:
                print(f'sweep')
                jid = os.environ['SLURM_JOBID']
                wandb.init()
                hpo_config = wandb.config
                self.params.update_params(hpo_config)
                logging.info('HPO sweep %s, job ID %d, trial params:'%(self.sweep_id, jid))
                logging.info(self.params.log())
            else:
                exp_dir = os.path.join(*[self.root_dir, 'expts', self.config])
                ckpt_dir = os.path.join(exp_dir, 'checkpoints/')
                
                if not os.path.isdir(exp_dir):
                    print(f'making base dir: {exp_dir}')
                    os.makedirs(exp_dir)
                    os.makedirs(os.path.join(exp_dir, 'checkpoints/'))
                elif not os.path.isdir(ckpt_dir):
                    print(f'making ckpt dir: {ckpt_dir}')
                    os.makedirs(os.path.join(exp_dir, 'checkpoints/'))
                else:
                    print(f'dir exists: {ckpt_dir}')
                
                if not os.path.isdir(exp_dir):
                    print(f'making dir: {exp_dir}')
                    os.makedirs(exp_dir)
                    os.makedirs(os.path.join(exp_dir, 'checkpoints/'))
                    
                self.params.experiment_dir = os.path.abspath(exp_dir)
                self.params.checkpoint_path = os.path.join(exp_dir, 'checkpoints/ckpt.tar')
                self.params.gen_ckpt_path = os.path.join(exp_dir, 'checkpoints/Nyx-TSIT/NyxG.pt')
                self.params.resuming = True if os.path.isfile(self.params.checkpoint_path) else False
                wandb.init(config=self.params.params, name=self.params.name, project=self.params.project, entity=self.params.entity, resume=self.params.resuming, dir=self.params.experiment_dir)
                #wandb.init(config=self.params.params, name=self.params.name, project=self.params.project, entity=self.params.entity, resume=self.params.resuming)

        # setup output dir
        if self.sweep_id:
            exp_dir = os.path.join(*[self.root_dir, 'sweeps', self.sweep_id, self.config, jid])
        else:
            exp_dir = os.path.join(*[self.root_dir, 'expts', self.config])
            ckpt_dir = os.path.join(exp_dir, 'checkpoints/')
        if self.world_rank==0:
            print(f'checking dir: {ckpt_dir}')
            if not os.path.isdir(exp_dir):
                print(f'making dir: {exp_dir}')
                os.makedirs(exp_dir)
                os.makedirs(os.path.join(exp_dir, 'checkpoints/'))
            elif not os.path.isdir(ckpt_dir):
                print(f'making dir: {ckpt_dir}')
                os.makedirs(os.path.join(exp_dir, 'checkpoints/'))
            else:
                print(f'dir exists: {ckpt_dir}')

        self.params.experiment_dir = os.path.abspath(exp_dir)
        self.params.checkpoint_path = os.path.join(exp_dir, 'checkpoints/ckpt.tar')
        self.params.resuming = True if os.path.isfile(self.params.checkpoint_path) else False

        if self.sweep_id and dist.is_initialized():
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            assert self.world_rank == rank
            if rank != 0: 
                self.params = None
            # Broadcast sweep config -- after here params should be finalized
            self.params = comm.bcast(self.params, root=0)
            
        world_rank = torch.distributed.get_rank()
        local_rank = int(os.environ['LOCAL_RANK'])
        
        self.params.distributed = False
        if 'WORLD_SIZE' in os.environ:
            self.params.distributed = int(os.environ['WORLD_SIZE']) > 1
            world_size = int(os.environ['WORLD_SIZE'])
        else:
            world_size = 1
        
        torch.backends.cudnn.benchmark = True
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda:%d'%local_rank)

        if self.world_rank == 0:
            hparams = ruamelDict()
            yaml = YAML()
            for key, value in self.params.params.items():
                hparams[str(key)] = str(value)
            with open(os.path.join(self.params.experiment_dir, 'hyperparams.yaml'), 'w') as hpfile:
                yaml.dump(hparams,  hpfile)

        self.params.global_batch_size = self.params.batch_size
        self.params.local_batch_size = int(self.params.batch_size//self.world_size)

        logging.info('rank %d, begin data loader init'%self.world_rank)
        '''
        self.train_data_loader, self.train_dataset, self.train_sampler = get_data_loader(self.params, distributed=dist.is_initialized(), train=True)
        self.valid_data_loader, self.valid_dataset, self.valid_sampler = get_data_loader(self.params, distributed=dist.is_initialized(), train=False)
        '''
        self.train_data_loader, self.valid_data_loader = get_data_loader_distributed(self.params, self.world_rank, device.index)
        logging.info('rank %d, data loader initialized'%self.world_rank)

        self.pix2pix_model = Pix2PixModel(self.params, dist.is_initialized(), self.local_rank, self.device)

        self.generated = None
        self.optimizerG, self.optimizerD = self.pix2pix_model.create_optimizers(self.params)
        # constant, then linear LR decay: chain schedules together
        constG = lr_scheduler.ConstantLR(self.optimizerG, factor=1., total_iters=self.params.niter)
        constD = lr_scheduler.ConstantLR(self.optimizerD, factor=1., total_iters=self.params.niter)
        linearG = lr_scheduler.LinearLR(self.optimizerG, start_factor=1., end_factor=0., total_iters=self.params.niter_decay)
        linearD = lr_scheduler.LinearLR(self.optimizerD, start_factor=1., end_factor=0., total_iters=self.params.niter_decay)
        self.schedulerG = lr_scheduler.SequentialLR(self.optimizerG, schedulers=[constG, linearG], milestones=[self.params.niter])
        self.schedulerD = lr_scheduler.SequentialLR(self.optimizerD, schedulers=[constD, linearD], milestones=[self.params.niter])

        if self.params.amp:
            self.grad_scaler = torch.cuda.amp.GradScaler()
        
        self.iters = 0
        self.startEpoch = 0

        if self.params.resuming:
            logging.info("Loading checkpoint %s"%self.params.checkpoint_path)
            self.restore_checkpoint(self.params.checkpoint_path)

        self.epoch = self.startEpoch

        self.logs = {}

        # launch training
        self.train()
        
    def train(self):
        if self.log_to_screen:
            logging.info("Starting Training Loop...")
        best = np.inf
        for epoch in range(self.startEpoch, self.params.niter+self.params.niter_decay):
            self.epoch = epoch
            '''
            if dist.is_initialized():
                # shuffles data before every epoch
                self.train_sampler.set_epoch(epoch)
                self.valid_sampler.set_epoch(epoch)
            '''
            
            start = time.time()
            tr_time = self.train_one_epoch()
            valid_time, fields = self.validate_one_epoch()
            #print(f'fields = {len(fields)}')
            self.schedulerG.step()
            self.schedulerD.step()

            is_best = self.logs['acc'] >= best
            best = max(self.logs['acc'], best)

            if self.world_rank == 0:
                if self.params.save_checkpoint:
                    print('saving checkpoint...')
                    #checkpoint at the end of every epoch
                    self.save_checkpoint(self.params.checkpoint_path, is_best=is_best)

            if self.log_to_wandb:
                fig = viz_fields(fields, self.params.output)
                self.logs['viz'] = wandb.Image(fig)
                plt.close(fig)
                self.logs['learning_rate_G'] = self.optimizerG.param_groups[0]['lr']
                wandb.log(self.logs, step=self.epoch+1)

            if self.log_to_screen:
                logging.info('Time taken for epoch {} is {} sec'.format(self.epoch+1, time.time()-start))
                logging.info('Train time = {}, Valid time = {}'.format(tr_time, valid_time))
                logging.info('G losses = '+str(self.g_losses))
                logging.info('D losses = '+str(self.d_losses))
                #logging.info('ACC = %f'%self.logs['acc'])
                
        if self.log_to_wandb:
            wandb.finish()

    def train_one_epoch(self):
        tr_time = 0
        self.pix2pix_model.set_train()
        batch_size = self.params.local_batch_size # batch size per gpu

        tr_start = time.time()
        g_time = 0.
        d_time = 0.
        data_time = 0.
        for i, (image, target) in enumerate(self.train_data_loader, 0):
            
            self.iters += 1
            timer = time.time()
            data = (image.to(self.device), target.to(self.device))
            data_time += time.time() - timer
            self.pix2pix_model.zero_all_grad()
            
            #print(f'image = {data[0].size()}')
            #print(f'target = {data[1].size()}')

            # Training
            # train generator
            timer = time.time()
            self.run_generator_one_step(data)
            if self.world_rank ==0 and i%16 ==0: print(f'generator made step {i}')
            g_time += time.time() - timer
            timer = time.time()
            self.run_discriminator_one_step(data)
            if self.world_rank ==0 and i%16 ==0: print(f'discriminator made step {i}')
            d_time += time.time() - timer
        
        tr_time = time.time() - tr_start
        #print(f'Rank {self.world_rank} made one pass')
        
        if self.log_to_screen: logging.info('Total=%f, G=%f, D=%f, data=%f, next=%f'%(tr_time, g_time, d_time, data_time, tr_time - (g_time+ d_time + data_time)))
        self.logs =  {**self.g_losses, **self.d_losses} 

        if dist.is_initialized():
            for key in self.logs.keys():
                if not torch.is_tensor(self.logs[key]):
                    continue
                dist.all_reduce(self.logs[key].detach())
                self.logs[key] = float(self.logs[key]/dist.get_world_size())

        return tr_time

    
    def validate_one_epoch(self):
        self.pix2pix_model.set_eval()
        valid_start = time.time()
        preds = []
        targets = []
        acc = []
        inps = []
        nc, iw ,ih = self.params.output_nc, self.params.img_size[0], self.params.img_size[1]
        loop = time.time()
        acctime = 0.
        g_time = 0.
        data_time = 0.
        with torch.no_grad():
            for idx, (image, target) in enumerate(self.valid_data_loader):
                timer = time.time()
                data = (image.to(self.device), target.to(self.device))
                data_time += time.time() - timer
                timer = time.time()
                gen = self.generate_validation(data)
                g_time += time.time() - timer
                timer = time.time()
                acc.append(weighted_acc_torch_channels(unlog_tp_torch(gen, self.params.precip_eps), 
                                                       unlog_tp_torch(data[1], self.params.precip_eps)))
                acctime += time.time() - timer
                preds.append(gen.detach())
                targets.append(data[1].detach())
                inps.append(image.detach())

        timer = time.time()
        preds = torch.cat(preds)
        targets = torch.cat(targets)
        acc = torch.cat(acc)
        inps = torch.cat(inps)
        '''
        # All-gather for full validation set currently OOMs
        if self.world_size > 1:
            # gather the sizes
            sz = torch.tensor(preds.shape[0]).float().to(self.device)
            sz_gather = [torch.zeros((1,)).float().to(self.device) for _ in range(self.world_size)]
            dist.all_gather(sz_gather, sz)
            # gather all the preds 
            preds_global = [torch.zeros(int(sz_loc.item()), nc, ih, iw).float().to(self.device) for sz_loc in sz_gather]
            dist.all_gather(preds_global, preds)
            preds = torch.cat([x for x in preds_global])
            targets_global = [torch.zeros(int(sz_loc.item()), nc, ih, iw).float().to(self.device) for sz_loc in sz_gather]
            dist.all_gather(targets_global, targets)
            targets = torch.cat([x for x in targets_global])
            acc_global = [torch.zeros(int(sz_loc.item()), nc).float().to(self.device) for sz_loc in sz_gather]
            dist.all_gather(acc_global, acc)
            acc = torch.cat([x for x in acc_global])
        '''
        sample_idx = np.random.randint(max(preds.size()[0], targets.size()[0]))
        fields = [preds[sample_idx].detach().cpu().numpy(), targets[sample_idx].detach().cpu().numpy()]

        valid_time = time.time() - valid_start
        self.logs.update(
                {'acc': acc.mean().item(),
                }
        )
        agg = time.time() - timer 
        if self.log_to_screen: logging.info('Total=%f, G=%f, data=%f, acc=%f, agg=%f, next=%f'%(valid_time, g_time, data_time, acctime, agg, valid_time - (g_time+ data_time + acctime + agg)))
        return valid_time, fields

    def save_checkpoint(self, checkpoint_path, is_best=False, model=None):
        if not model:
            model = self.pix2pix_model
        torch.save({'iters': self.iters, 'epoch': self.epoch, 
                    'model_state_G': model.save_state('generator'), 'model_state_D': model.save_state('discriminator'), 'model_state_E': model.save_state('encoder'),
                    'optimizerG_state_dict': self.optimizerG.state_dict(), 'schedulerG_state_dict': self.schedulerG.state_dict(),
                    'optimizerD_state_dict': self.optimizerD.state_dict(), 'schedulerD_state_dict': self.schedulerD.state_dict()},
                   checkpoint_path)
        if is_best:
            torch.save({'iters': self.iters, 'epoch': self.epoch, 
                        'model_state_G': model.save_state('generator'), 'model_state_D': model.save_state('discriminator'), 'model_state_E': model.save_state('encoder'),
                        'optimizerG_state_dict': self.optimizerG.state_dict(), 'schedulerG_state_dict': self.schedulerG.state_dict(),
                        'optimizerD_state_dict': self.optimizerD.state_dict(), 'schedulerD_state_dict': self.schedulerD.state_dict()},
                       checkpoint_path.replace('.tar', '_best.tar'))
            
            '''
            netG_scripted = torch.jit.script(model.netG.module) # Export to TorchScript
            netG_scripted.save(self.params.gen_ckpt_path) # Save
            '''
            
            print('SAVED NEW CHECKPOINT ##############################################')
        '''
        else:
            if not exists(self.params.gen_ckpt_path):
                netG_scripted = torch.jit.script(model.netG.module) # Export to TorchScript
                netG_scripted.save(self.params.gen_ckpt_path) # Save
                
                print('Is not best but saved generator anyway.')
        '''
            
    def restore_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cuda:{}'.format(self.local_rank))
        self.pix2pix_model.load_state(checkpoint)
        self.iters = checkpoint['iters']
        self.startEpoch = checkpoint['epoch'] + 1
        self.optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
        self.optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
        self.schedulerG.load_state_dict(checkpoint['schedulerG_state_dict'])
        self.schedulerD.load_state_dict(checkpoint['schedulerD_state_dict'])

    def run_generator_one_step(self, data):
        self.optimizerG.zero_grad()
        with torch.cuda.amp.autocast(self.params.amp):
            g_losses, generated = self.pix2pix_model.compute_generator_loss(data[0], data[1],self.epoch)
            g_loss = sum(g_losses.values()).mean()

            self.g_losses = {k: v.item() for k,v in g_losses.items()}
            self.generated = generated
            if self.params.amp:
                self.grad_scaler.scale(g_loss).backward()
                self.grad_scaler.step(self.optimizerG)
                self.grad_scaler.update()
            else:
                g_loss.backward()
                self.optimizerG.step()

    def run_discriminator_one_step(self, data):
        self.optimizerD.zero_grad()
        with torch.cuda.amp.autocast(self.params.amp):
            d_losses = self.pix2pix_model.compute_discriminator_loss(data[0], data[1])
            d_loss = sum(d_losses.values()).mean()
            self.d_losses = {k: v.item() for k,v in d_losses.items()}
            if self.params.amp:
                self.grad_scaler.scale(d_loss).backward()
                self.grad_scaler.step(self.optimizerG)
                self.grad_scaler.update()
            else:
                d_loss.backward()
                self.optimizerD.step()


    def get_latest_generated(self):
        return self.generated

    def generate_validation(self, data):
        generated, _ = self.pix2pix_model.generate_fake(data[0], data[1])
        return generated

