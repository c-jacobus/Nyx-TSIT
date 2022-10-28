import numpy as np
import h5py
import torch
import torch.nn.functional as nn
from torch.nn.parallel import DistributedDataParallel
from utils.YParams import YParams
from models.pix2pix_model import Pix2PixModel
import models.networks as networks
from utils.YParams import YParams

import argparse
import resource
import os
import h5py
from collections import OrderedDict
import datetime
import time
import sys
import logging
from os.path import exists

'''
This performs chunk-wise inference on a given file using a given config.

Note: the width and trim is such that each chunk is 192 wide,
The data_size set for the used config in config.yaml must be 192.  
'''

parser = argparse.ArgumentParser()
parser.add_argument("--folder", default='/pscratch/sd/c/cjacobus/ML_Hydro_train/logs/', type=str) # where logs/ckpts are saved

parser.add_argument("--config", default='valid_noise', type=str) # .yaml config name

parser.add_argument("--subsec", default='16GPU/00/', type=str) # train has these saved like this by GPU count
parser.add_argument("--weights", default='training_checkpoints/ckpt.tar', type=str) # as per train defination
parser.add_argument("--yaml_config", default='./config/tsit.yaml', type=str) # .yaml name
parser.add_argument("--datapath", default='/pscratch/sd/c/cjacobus/Nyx_TSIT_train/valid_s2_512_pyr05_hydro.h5', type=str) # file to do inference on
parser.add_argument("--realpath", default='/pscratch/sd/c/cjacobus/Nyx_TSIT_train/train_s1_512_pyr05_hydro.h5', type=str) # file to do inference on
parser.add_argument("--trim", default=32, type=int) # width of "crust" to trim off the inside faced of inferred chunks
parser.add_argument("--size", default=128, type=int) # width to keep of inferred chunks 
parser.add_argument("--full_dim", default=512, type=int) # width of total field
parser.add_argument("--flavor", default='flux', type=str) # OLD, if use attention model
parser.add_argument("--dummy", default=False, type=bool) # for debugging
parser.add_argument("--skip", default=False, type=bool) # just infer one chunk
parser.add_argument("--native", default=True, type=bool) # infer native?
parser.add_argument("--derived", default=False, type=bool) # infer derived?
parser.add_argument("--flux", default=True, type=bool) # model trained on flux or tau
parser.add_argument("--template", default=True, type=bool) # whether or not to overwrite a template file, rather than making a new one
parser.add_argument("--temp_path", default='/pscratch/sd/c/cjacobus/valid_noise/infer1.hdf5', type=str)
args = parser.parse_args()

from mpi4py import MPI
world_size = MPI.COMM_WORLD.Get_size()
world_rank = MPI.COMM_WORLD.Get_rank()
name = MPI.Get_processor_name()
local_rank = int(os.environ['SLURM_LOCALID'])
device = torch.device('cuda:%d'%local_rank)

size=args.size
trim=args.trim
full_dim=args.full_dim
dtype=np.single


params = YParams(os.path.abspath(args.yaml_config), args.config)
params.data_size = 192
exp_dir = os.path.join('/pscratch/sd/c/cjacobus/tsit/expts/', args.config)
params.experiment_dir = os.path.abspath(exp_dir)
params.checkpoint_path = os.path.join(exp_dir, 'checkpoints/ckpt.tar')

if  world_rank==0:
    print("checkpoint path: {}".format(params.checkpoint_path))
    print("Initializing model...")
    print("Config: {}, {}".format(args.yaml_config, args.config))

netG = networks.define_G(params).to(device)
   
if world_rank==0:
    print("Initialized model [✓]")
    print("Loading checkpoint: {}".format(params.checkpoint_path))

checkpoint = torch.load(params.checkpoint_path, map_location='cuda:%d'%local_rank)

new_model_state = OrderedDict()
model_key = 'model_state_G' if 'model_state_G' in checkpoint else 'state_dict'
for key in checkpoint[model_key].keys():
    if 'module.' in key: # model was stored using ddp which prepends module
        name = str(key[7:])
        new_model_state[name] = checkpoint[model_key][key]
    else:
        new_model_state[key] = checkpoint[model_key][key]
    
netG.load_state_dict(new_model_state)

if world_rank==0: print("Loaded model checkpoint [✓]")

        
if not args.dummy:     
    if args.template:
        save_path = args.temp_path
    else: 
        save_name = "infer_{}_size_{}_trim_{}.h5".format(args.flavor,size,trim)
        save_path = os.path.join(folder_path, save_name)

    file_exists = exists(save_path)

    if  world_rank==0:
        print("Write path: {}".format(save_path))
        if file_exists: print("File already exists")
    
    with h5py.File(save_path, 'a', driver='mpio', comm=MPI.COMM_WORLD) as hf:
        
        if not args.template and not file_exists:
            hf.attrs['format'] = "nyx-lyaf"
            hf.attrs['chunk'] = size
            hf.attrs['trim'] = trim
            hf.attrs['flavor'] = args.flavor

            dom = hf.create_group("domain")
            dom.attrs['size'] = [80,80,80]
            dom.attrs['shape'] = [full_dim,full_dim,full_dim]

            uni = hf.create_group("universe")
            uni.attrs['hubble'] = 0.675
            uni.attrs['omega_b'] = 0.0487
            uni.attrs['omega_l'] = 0.69
            uni.attrs['omega_m'] = 0.31
            uni.attrs['redshift'] = 2.9999991588912964

            if args.native:
                # rho = hf.create_dataset("native_fields/baryon_density", data=np.exp(14.*final[0,0,:,:,:]))
                rho = hf.create_dataset("native_fields/baryon_density", (full_dim,full_dim,full_dim), dtype='<f4')
                rho.attrs['units'] = "(mean)"

                # vx = hf.create_dataset("native_fields/velocity_x", data=final[0,1,:,:,:]*9e7)
                vx = hf.create_dataset("native_fields/velocity_x", (full_dim,full_dim,full_dim), dtype='<f4')
                vx.attrs['units'] = "km/s"

                # vy = hf.create_dataset("native_fields/velocity_y", data=final[0,2,:,:,:]*9e7)
                vy = hf.create_dataset("native_fields/velocity_y", (full_dim,full_dim,full_dim), dtype='<f4')
                vy.attrs['units'] = "km/s"

                # vz = hf.create_dataset("native_fields/velocity_z", data=final[0,3,:,:,:]*9e7)
                vz = hf.create_dataset("native_fields/velocity_z", (full_dim,full_dim,full_dim), dtype='<f4')
                vz.attrs['units'] = "km/s"

                # temp = hf.create_dataset("native_fields/temperature", data=np.exp(8.*(final[0,4,:,:,:] + 1.5)))
                temp = hf.create_dataset("native_fields/temperature", (full_dim,full_dim,full_dim), dtype='<f4')
                temp.attrs['units'] = "K"

            if args.derived:
                tau = hf.create_dataset("derived_fields/tau_red", (full_dim,full_dim,full_dim), dtype='<f4')
                tau.attrs['units'] = "(none)"
                
        if args.derived:
            flux = hf.create_dataset("derived_fields/flux_red", (full_dim,full_dim,full_dim), dtype='<f4')
            flux.attrs['units'] = "(none)"

        print("Rank {} initialized file".format(world_rank))
        
        f = h5py.File(args.datapath, 'r')
        rf = h5py.File(args.realpath, 'r')
        
        if  world_rank==0: print("Input data path: {}".format(args.datapath))
        full_dim = f['coarse'][0,0,0,:].shape[0]
        if  world_rank==0: print("Dimension: {}".format(full_dim))
        slices=int(full_dim/size)
        
        hf.create_dataset("chunk_complete", (slices,slices,slices), dtype='<f4')
    
        if  world_rank==0:
            print("Dimension: {}".format(full_dim))
            print("Trimmed chunk size: {}".format(size))
            print("Trim: {}".format(trim))
            print("Slices: {}".format(slices))
            print("Sending {} chunks individually...".format(slices**3))
        
        
        time.sleep(world_rank*0.2)
        
        if not args.dummy: 
            for x in range(slices):
                x1 = int(x*size)
                x1_edge = 0 if (x<=0) else (trim*2 if (x>=slices-1) else trim)  
                x2 = int(min((x+1)*size,full_dim))
                x2_edge = 0 if (x>=slices-1) else (trim*2 if (x<=0) else trim)                            
                for y in range(slices):
                    y1 = int(y*size)
                    y1_edge = 0 if (y<=0) else (trim*2 if (y>=slices-1) else trim)
                    y2 = int(min((y+1)*size,full_dim))
                    y2_edge = 0 if (y>=slices-1) else (trim*2 if (y<=0) else trim) 
                    for z in range(slices):
                        z1 = int(z*size)
                        z1_edge = 0 if (z<=0) else (trim*2 if (z>=slices-1) else trim)
                        z2 = int(min((z+1)*size,full_dim))
                        z2_edge = 0 if (z>=slices-1) else (trim*2 if (z<=0) else trim) 

                        if not hf['chunk_complete'][x,y,z] == 1:
                            time.sleep(world_rank*0.2)

                            if not hf['chunk_complete'][x,y,z] == 1:
                                hf['chunk_complete'][x,y,z] = 1
                                start = time.perf_counter()

                                x_plus = None if (x2_edge == 0) else -x2_edge
                                y_plus = None if (y2_edge == 0) else -y2_edge
                                z_plus = None if (z2_edge == 0) else -z2_edge

                                sliced_in = f['coarse'][:, x1-x1_edge:x2+x2_edge, y1-y1_edge:y2+y2_edge, z1-z1_edge:z2+z2_edge].astype(dtype)
                                sliced_in = np.expand_dims(sliced_in, axis=0) # add batch dim
                                sliced_in = torch.from_numpy(sliced_in).to(device)
                                
                                sliced_in_real = rf['fine'][:, x1-x1_edge:x2+x2_edge, y1-y1_edge:y2+y2_edge, z1-z1_edge:z2+z2_edge].astype(dtype)
                                sliced_in_real = np.expand_dims(sliced_in_real, axis=0) # add batch dim
                                sliced_in_real = torch.from_numpy(sliced_in_real).to(device)

                                print("Rank {} received chunk [{},{},{}], input shape: {}".format(world_rank,x,y,z,sliced_in.shape))

                                with torch.no_grad():
                                    #chunk = model(sliced_in)
                                    chunk = netG(sliced_in, sliced_in_real, z=None)
                                    
                                print("Rank {} inferred chunk [{},{},{}], pred shape: {}".format(world_rank,x,y,z,chunk.shape))
                                
                                chunk = chunk.cpu()
                                chunk = chunk.numpy()
                                #chunk = np.minimum(chunk,1)
                                #chunk = np.maximum(chunk,-1)
                                    
                                

                                # un-normalize the NN output and write to file
                                if args.native:
                                    
                                    hf['native_fields']['baryon_density'][x1:x2,y1:y2,z1:z2] = np.exp(14.*chunk[0, 0, x1_edge:x_plus, y1_edge:y_plus, z1_edge:z_plus])
                                    hf['native_fields']['velocity_x'][x1:x2,y1:y2,z1:z2] = chunk[0, 1, x1_edge:x_plus, y1_edge:y_plus, z1_edge:z_plus]*9e7
                                    hf['native_fields']['velocity_y'][x1:x2,y1:y2,z1:z2] = chunk[0, 2, x1_edge:x_plus, y1_edge:y_plus, z1_edge:z_plus]*9e7
                                    hf['native_fields']['velocity_z'][x1:x2,y1:y2,z1:z2] = chunk[0, 3, x1_edge:x_plus, y1_edge:y_plus, z1_edge:z_plus]*9e7
                                    hf['native_fields']['temperature'][x1:x2,y1:y2,z1:z2] = np.exp(8.*(chunk[0, 4, x1_edge:x_plus, y1_edge:y_plus, z1_edge:z_plus] + 1.5))
                                    
                                    '''
                                    hf['native_fields']['baryon_density'][x1:x2,y1:y2,z1:z2] = np.exp(6.*(chunk[0, 0, x1_edge:x_plus, y1_edge:y_plus, z1_edge:z_plus] + 0.4))
                                    hf['native_fields']['velocity_x'][x1:x2,y1:y2,z1:z2] = chunk[0, 1, x1_edge:x_plus, y1_edge:y_plus, z1_edge:z_plus]*8e7
                                    hf['native_fields']['velocity_y'][x1:x2,y1:y2,z1:z2] = chunk[0, 2, x1_edge:x_plus, y1_edge:y_plus, z1_edge:z_plus]*8e7
                                    hf['native_fields']['velocity_z'][x1:x2,y1:y2,z1:z2] = chunk[0, 3, x1_edge:x_plus, y1_edge:y_plus, z1_edge:z_plus]*8e7
                                    hf['native_fields']['temperature'][x1:x2,y1:y2,z1:z2] = np.exp(4.*(chunk[0, 4, x1_edge:x_plus, y1_edge:y_plus, z1_edge:z_plus] + 3.0))
                                    '''
                                if args.derived:
                                    trimmed = chunk[0, 0, x1_edge:x_plus, y1_edge:y_plus, z1_edge:z_plus]
                                    if args.flux:
                                        trimmed = np.minimum(trimmed, 1)
                                        trimmed = np.maximum(trimmed, 1e-100)
                                        hf['derived_fields']['tau_red'][x1:x2,y1:y2,z1:z2] = -np.log(trimmed)
                                        hf['derived_fields']['flux_red'][x1:x2,y1:y2,z1:z2] = trimmed
                                    else:
                                        hf['derived_fields']['tau_red'][x1:x2,y1:y2,z1:z2] = tau = np.exp(10.*(trimmed + 0.5))
                                        hf['derived_fields']['flux_red'][x1:x2,y1:y2,z1:z2] = np.exp(-tau)
                                    
                                stop = time.perf_counter()
                                print("Rank {} wrote chunk [{},{},{}] to file, took {}s".format(world_rank,x,y,z, stop-start))
                                #print("Chunk [{},{},{}] output shape: {}".format(x,y,z,chunk.shape))

                        if args.skip: break
                    if args.skip: break
                if args.skip: break
    
        hf.close()
    
maxRSS = resource.getrusage(resource.RUSAGE_SELF)[2]

if  world_rank==0:
    print("MaxRSS: {} [GB]".format(maxRSS/1e6))

    print('DONE')