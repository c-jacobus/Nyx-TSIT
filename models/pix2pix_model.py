import torch
import torch.nn as nn
import torch.nn.functional as F
import models.networks as networks
from torch.nn.parallel import DistributedDataParallel


class Pix2PixModel():
    """
    Wrapper class for generator, discriminator, encoder models and related infrastructure.
    """
    def count_parameters(self):
        nG = sum(p.numel() for p in self.netG.parameters() if p.requires_grad)
        nD = sum(p.numel() for p in self.netD.parameters() if p.requires_grad)
        return {'Generator':nG, 'Discriminator':nD}

    def __init__(self, params, distributed, local_rank, device, isTrain=True):
        self.params = params
        self.FloatTensor = torch.cuda.FloatTensor
        self.device = device
        self.isTrain = isTrain

        self.netG, self.netD, self.netE = self.initialize_networks(params)
        if distributed:
            self.netG = nn.SyncBatchNorm.convert_sync_batchnorm(self.netG)
            self.netD = nn.SyncBatchNorm.convert_sync_batchnorm(self.netD) if self.netD else None
            self.netE = nn.SyncBatchNorm.convert_sync_batchnorm(self.netE) if self.netE else None
            ddp_args = {'device_ids':[local_rank], 'output_device':local_rank}
            self.netG = DistributedDataParallel(self.netG, **ddp_args, find_unused_parameters=True)
            self.netD = DistributedDataParallel(self.netD, **ddp_args, find_unused_parameters=True) if self.netD else None
            self.netE = DistributedDataParallel(self.netE, **ddp_args, find_unused_parameters=True) if self.netE else None

        # set loss functions
        if self.isTrain:
            self.criterionGAN = networks.GANLoss(
                params.gan_mode, tensor=self.FloatTensor, params=params)
            self.criterionFeat = torch.nn.L1Loss()
            self.L2 = torch.nn.MSELoss()
            if params.use_vae:
                self.KLDLoss = networks.KLDLoss()


    def set_train(self):
        self.netG.train()
        if self.netD: self.netD.train()
        if self.netE: self.netE.train()

    def set_eval(self):
        self.netG.eval()
        if self.netD: self.netD.eval()
        if self.netE: self.netE.eval()

    def zero_all_grad(self):
        self.netG.zero_grad()
        if self.netD: self.netD.zero_grad()
        if self.netE: self.netE.zero_grad()

    def save_state(self, net):
        if net=='generator':
            return self.netG.state_dict()
        elif  net=='discriminator':
            return self.netD.state_dict() if self.netD else None
        elif net=='encoder':
            return self.netE.state_dict() if self.netE else None

    def load_state(self, ckpt):
        self.netG.load_state_dict(ckpt['model_state_G'])
        if self.netD: self.netD.load_state_dict(ckpt['model_state_D'])
        if self.netE: self.netE.load_state_dict(ckpt['model_state_E'])


    """
    # Entry point for all calls involving forward pass
    # of deep networks. Branch to different
    # routines based on |mode|.
    def forward(self, data, mode):
        input_image, real_image = data
        if mode == 'generator':
            g_loss, generated = self.compute_generator_loss(
                input_image, real_image)
            return g_loss, generated
        elif mode == 'discriminator':
            d_loss = self.compute_discriminator_loss(
                input_image, real_image)
            return d_loss
        elif mode == 'encode_only':
            z, mu, logvar = self.encode_z(real_image)
            return mu, logvar
        elif mode == 'inference':
            with torch.no_grad():
                fake_image, _ = self.generate_fake(input_image, real_image)
            return fake_image
        else:
            raise ValueError("|mode| is invalid")
    """

    def create_optimizers(self, params):
        G_params = list(self.netG.parameters())
        if params.use_vae:
            G_params += list(self.netE.parameters())
        if self.isTrain:
            D_params = list(self.netD.parameters())

        if params.no_TTUR:
            beta1, beta2 = params.beta1, params.beta2
            G_lr, D_lr = params.lr, params.lr
        else:
            beta1, beta2 = 0, 0.9
            G_lr, D_lr = params.lr / 2, params.lr * 2

        optimizer_G = torch.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2))
        optimizer_D = torch.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2))

        return optimizer_G, optimizer_D

    
    def initialize_networks(self, params):
        netG = networks.define_G(params).to(self.device)
        netD = networks.define_D(params).to(self.device) if self.isTrain else None
        netE = networks.define_E(params).to(self.device) if params.use_vae else None

        return netG, netD, netE
    

    def compute_generator_loss(self, input_image, real_image, epoch):
        G_losses = {}
        '''
        if self.params.output == 'dens':
            real_image = real_image[0,:,:,:] 
        '''
        fake_image, KLD_loss = self.generate_fake(
            input_image, real_image, compute_kld_loss=self.params.use_vae)

        if self.params.use_vae:
            G_losses['KLD'] = KLD_loss

        if self.params.cat_inp:
            pred_fake, pred_real = self.discriminate(fake_image, real_image, input_image)
        else:
            pred_fake, pred_real = self.discriminate(fake_image, real_image)

        G_losses['GAN'] = self.criterionGAN(pred_fake, True,
                                            for_discriminator=False)

        if not self.params.no_ganFeat_loss:
            num_D = len(pred_fake)
            GAN_Feat_loss = self.FloatTensor(1).fill_(0)
            for i in range(num_D):  # for each discriminator
                # last output is the final prediction, so we exclude it
                num_intermediate_outputs = len(pred_fake[i]) - 1
                for j in range(num_intermediate_outputs):  # for each layer output
                    unweighted_loss = self.criterionFeat(
                        pred_fake[i][j], pred_real[i][j].detach())
                    GAN_Feat_loss += unweighted_loss * self.params.lambda_feat / num_D
            G_losses['GAN_Feat'] = GAN_Feat_loss

        if self.params.use_spec_loss:
            G_losses['Spec'] = self.compute_spec_loss(fake_image, real_image)
        if self.params.use_l1_loss:
            G_losses['L1'] = (self.params.lambda_l1+(epoch*self.params.L1_schedule))*self.criterionFeat(fake_image, real_image)

        return G_losses, fake_image

    def compute_discriminator_loss(self, input_image, real_image):
        D_losses = {}
        with torch.no_grad():
            fake_image, _ = self.generate_fake(input_image, real_image)
            fake_image = fake_image.detach()
            fake_image.requires_grad_()

        if self.params.cat_inp:
            pred_fake, pred_real = self.discriminate(fake_image, real_image, input_image)
        else:
            pred_fake, pred_real = self.discriminate(fake_image, real_image)

        D_losses['D_Fake'] = self.criterionGAN(pred_fake, False,
                                               for_discriminator=True)
        D_losses['D_real'] = self.criterionGAN(pred_real, True,
                                               for_discriminator=True)

        return D_losses

    def encode_z(self, real_image):
        mu, logvar = self.netE(real_image)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


    def compute_spec_loss(self, fake_image, style):
        # Take FFT and return L1 loss
        
        fake_image_norm = torch.clone(fake_image)
        #print(fake_image_norm.size())

        style_norm = torch.clone(style)
        
        if self.params.output_nc == 5:
            fake_image_norm[0,0,:,:,:] = torch.exp(14.*fake_image_norm[0,0,:,:,:])
            fake_image_norm[0,1,:,:,:] = fake_image_norm[0,1,:,:,:]*9e7
            fake_image_norm[0,2,:,:,:] = fake_image_norm[0,2,:,:,:]*9e7
            fake_image_norm[0,3,:,:,:] = fake_image_norm[0,3,:,:,:]*9e7
            fake_image_norm[0,4,:,:,:] = torch.exp(8.*fake_image_norm[0,4,:,:,:] + 1.5)

            style_norm[0,0,:,:,:] = torch.exp(14.*style_norm[0,0,:,:,:])
            style_norm[0,1,:,:,:] = style_norm[0,1,:,:,:]*9e7
            style_norm[0,2,:,:,:] = style_norm[0,2,:,:,:]*9e7
            style_norm[0,3,:,:,:] = style_norm[0,3,:,:,:]*9e7
            style_norm[0,4,:,:,:] = torch.exp(8.*style_norm[0,4,:,:,:] + 1.5)
            
        if self.params.output == 'dens' and False:
            fake_image_norm[0,0,:,:,:] = torch.exp(14.*fake_image_norm[0,0,:,:,:])
            style_norm[0,0,:,:,:] = torch.exp(14.*style_norm[0,0,:,:,:])
            
        if self.params.output == 'dual' and False:
            fake_image_norm[0,0,:,:,:] = torch.exp(14.*fake_image_norm[0,0,:,:,:])
            fake_image_norm[0,1,:,:,:] = torch.exp(8.*fake_image_norm[0,1,:,:,:] + 1.5)
            style_norm[0,0,:,:,:] = torch.exp(14.*style_norm[0,0,:,:,:])
            style_norm[0,1,:,:,:] = torch.exp(8.*style_norm[0,1,:,:,:] + 1.5)
            
              
        
        #fake_fft = torch.fft.rfftn(fake_image_norm, norm='ortho', dim=(-1,-2,-3)) #[:,:,:-64,:-64,:-64]
        
        fake_fft = torch.fft.fftn(fake_image_norm, norm='ortho', dim=(-1,-2,-3))
        fake_fft = torch.fft.fftshift(fake_fft)[:,:,4:-4,4:-4,4:-4]
        
        #print(fake_fft.size())
        
        #true_fft = torch.fft.rfftn(style_norm, norm='ortho', dim=(-1,-2,-3)) #[:,:,:-64,:-64,:-64]
        
        true_fft = torch.fft.fftn(style_norm, norm='ortho', dim=(-1,-2,-3))
        true_fft = torch.fft.fftshift(true_fft)[:,:,4:-4,4:-4,4:-4]
        
        #unweighted_loss = self.criterionFeat(torch.log1p(fake_fft.abs()), torch.log1p(true_fft.abs()))
        #unweighted_loss = self.criterionFeat(torch.square(fake_fft.abs()), torch.square(true_fft.abs()))
        #unweighted_loss = self.criterionFeat(fake_fft.abs(), true_fft.abs())
        unweighted_loss = self.L2(torch.view_as_real(fake_fft), torch.view_as_real(true_fft))
        
        log1p_loss = self.L2(torch.log1p(torch.abs(torch.view_as_real(fake_fft))), torch.log1p(torch.abs(torch.view_as_real(true_fft))))
        
        return unweighted_loss*self.params.lambda_spec + log1p_loss*self.params.lambda_spec_2
        

    def generate_fake(self, input_image, real_image, compute_kld_loss=False):
        z = None
        KLD_loss = None
        if self.params.use_vae:
            z, mu, logvar = self.encode_z(real_image)
            if compute_kld_loss:
                KLD_loss = self.KLDLoss(mu, logvar) * self.params.lambda_kld

        fake_image = self.netG(input_image, real_image, z=z)

        assert (not compute_kld_loss) or self.params.use_vae, \
            "You cannot compute KLD loss if opt.use_vae == False"

        return fake_image, KLD_loss

    # Given fake and real image, return the prediction of discriminator
    # for each fake and real image. The condition is used in SIS.
    def discriminate(self, fake_image, real_image, condition=None):
        if self.params.cat_inp:
            assert condition is not None
            fake_concat = torch.cat([condition, fake_image], dim=1)
            real_concat = torch.cat([condition, real_image], dim=1)
        else:
            assert condition is None
            fake_concat = fake_image
            real_concat = real_image

        # In Batch Normalization, the fake and real images are
        # recommended to be in the same batch to avoid disparate
        # statistics in fake and real images.
        # So both fake and real images are fed to D all at once.
        fake_and_real = torch.cat([fake_concat, real_concat], dim=0)

        discriminator_out = self.netD(fake_and_real)

        pred_fake, pred_real = self.divide_pred(discriminator_out)

        return pred_fake, pred_real

    # Take the prediction of fake and real images from the combined batch
    def divide_pred(self, pred):
        # the prediction contains the intermediate outputs of multi-scale GAN,
        # so it's usually a list
        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                real.append([tensor[tensor.size(0) // 2:] for tensor in p])
        else:
            fake = pred[:pred.size(0) // 2]
            real = pred[pred.size(0) // 2:]

        return fake, real

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std) + mu
    
