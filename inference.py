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
import shutil

from scipy.ndimage import gaussian_filter 
import random



'''
This performs chunk-wise inference on a given file using a given config.

Note: the width and trim is such that each chunk is 192 wide,
The data_size set for the used config in config.yaml must be 192.  
'''

parser = argparse.ArgumentParser()
#parser.add_argument("--folder", default='/pscratch/sd/c/cjacobus/ML_Hydro_train/logs/', type=str) # where logs/ckpts are saved

#parser.add_argument("--config", default='pyr_07_DENS_tanh_512GANcatorig', type=str) # .yaml config name

parser.add_argument("--rho_config", default='pyr_07_DENS_tanh_512GANcatorig_L1', type=str) # .yaml config name  #pyr_07_DENS_tanh_512GANcatorig_L1    pyr_07_DENS_tanh_512GANcatorig    pyr_07_DENS_tanh_GAN_100initNoise_sqrt
parser.add_argument("--temp_config", default='pyr_half_TEMP_tanh_512GANcatorig_L1_GAN', type=str) # pyr_half_TEMP_tanh_512GANcatorig_L1_GAN    pyr_half_TEMP_superGAN_Noise_100init_sqrt   .yaml config name
parser.add_argument("--flux_config", default='spectral2', type=str) # .yaml config name  spectral2 Flux_80_spec_L1_initNoise

parser.add_argument("--subsec", default='16GPU/00/', type=str) # train has these saved like this by GPU count
parser.add_argument("--weights", default='training_checkpoints/ckpt.tar', type=str) # as per train defination
parser.add_argument("--yaml_config", default='./config/tsit.yaml', type=str) # .yaml name
parser.add_argument("--datapath", default='/pscratch/sd/c/cjacobus/Nyx_512/pyr_sub_mix_s1_DENS.h5', type=str) # file to do inference on
parser.add_argument("--realpath", default='/pscratch/sd/c/cjacobus/Nyx_512/pyr_sub_mix_s1_DENS.h5', type=str) 
parser.add_argument("--scaffold", default='/pscratch/sd/z/zarija/MLHydro/L80_N512_z3_s1.hdf5', type=str)
parser.add_argument("--trim", default=16, type=int) # width of "crust" to trim off the inside faced of inferred chunks
parser.add_argument("--size", default=32, type=int) # width to keep of inferred chunks 
parser.add_argument("--full_dim", default=512, type=int) # width of total field
parser.add_argument("--flavor", default='flux', type=str) # 
parser.add_argument("--dummy", default=False, type=bool) # for debugging
parser.add_argument("--skip", default=False, type=bool) # just infer one chunk
parser.add_argument("--native", default=True, type=bool) # infer native?
parser.add_argument("--derived", default=False, type=bool) # infer derived?
parser.add_argument("--flux", default=True, type=bool) # model trained on flux or tau
parser.add_argument("--output", default='dub', type=str) # 'dual' , 'flux', 'dens'
parser.add_argument("--template", default=True, type=bool) # whether or not to overwrite a template file, rather than making a new one
parser.add_argument("--make_copy", default=True, type=bool) 
parser.add_argument("--temp_path", default='/pscratch/sd/z/zarija/MLHydro/DUB_80_TEST_v0.hdf5', type=str)
parser.add_argument("--mask", default=True, type=bool) 
parser.add_argument("--rescale", default=True, type=bool) 
parser.add_argument("--blur", default=False, type=bool) 
parser.add_argument("--seed", default=0, type=int)
args = parser.parse_args()

in_size = args.size + 2*args.trim

seed = args.seed
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

from mpi4py import MPI
world_size = MPI.COMM_WORLD.Get_size()
world_rank = MPI.COMM_WORLD.Get_rank()
name = MPI.Get_processor_name()
local_rank = int(os.environ['SLURM_LOCALID'])
print(f'World Rank: {world_rank}, Local Rank: {local_rank}')
device = torch.device('cuda:%d'%local_rank)

size=args.size
trim=args.trim
full_dim=args.full_dim
dtype=np.single

#params = YParams(os.path.abspath(args.yaml_config), args.config)
#params.data_size = 128
if not args.output == 'flux':
    rho_params = YParams(os.path.abspath(args.yaml_config), args.rho_config)
    rho_params.data_size = in_size
    rho_exp_dir = os.path.join('/global/cfs/cdirs/m3900/cjacobus/expdir/expts/', args.rho_config)
    rho_params.experiment_dir = os.path.abspath(rho_exp_dir)
    rho_params.checkpoint_path = os.path.join(rho_exp_dir, 'checkpoints/ckpt.tar')

    if  world_rank==0:
        print("1st Checkpoint Path: {}".format(rho_params.checkpoint_path))
        print("Initializing 1st model...")
        print("1st Config: {}, {}".format(args.yaml_config, args.rho_config))

    #netG = networks.define_G(params).to(device)

    rho_netG = networks.define_G(rho_params).to(device)

    if world_rank==0:
        print("Initialized model [✓]")
        print("Loading Checkpoint: {}".format(rho_params.checkpoint_path))

    rho_checkpoint = torch.load(rho_params.checkpoint_path, map_location='cuda:%d'%local_rank)

    new_model_state = OrderedDict()
    model_key = 'model_state_G' if 'model_state_G' in rho_checkpoint else 'state_dict'
    for key in rho_checkpoint[model_key].keys():
        if 'module.' in key: # model was stored using ddp which prepends module
            name = str(key[7:])
            new_model_state[name] = rho_checkpoint[model_key][key]
        else:
            new_model_state[key] = rho_checkpoint[model_key][key]

    rho_netG.load_state_dict(new_model_state)

    if world_rank==0: print("Loaded model checkpoint [✓]")
    
    if world_rank==0: 
        print(f"1st Model Featmult:")
        for i, param in enumerate(rho_netG.noise_stream.featmult):
            print(f"Parameter {i} size : {param.size()}")
            print(f"Parameter {i} mean : {torch.mean(param)}")
    #########


    temp_params = YParams(os.path.abspath(args.yaml_config), args.temp_config)
    temp_params.data_size = in_size
    temp_exp_dir = os.path.join('/global/cfs/cdirs/m3900/cjacobus/expdir/expts/', args.temp_config)
    temp_params.experiment_dir = os.path.abspath(temp_exp_dir)
    temp_params.checkpoint_path = os.path.join(temp_exp_dir, 'checkpoints/ckpt.tar')

    if  world_rank==0:
        print("2nd Checkpoint Path: {}".format(temp_params.checkpoint_path))
        print("Initializing 2nd model...")
        print("2nd Config: {}, {}".format(args.yaml_config, args.temp_config))

    temp_netG = networks.define_G(temp_params).to(device)

    if world_rank==0:
        print("Initialized model [✓]")
        print("Loading Checkpoint: {}".format(temp_params.checkpoint_path))

    temp_checkpoint = torch.load(temp_params.checkpoint_path, map_location='cuda:%d'%local_rank)

    new_model_state = OrderedDict()
    model_key = 'model_state_G' if 'model_state_G' in temp_checkpoint else 'state_dict'
    for key in temp_checkpoint[model_key].keys():
        if 'module.' in key: # model was stored using ddp which prepends module
            name = str(key[7:])
            new_model_state[name] = temp_checkpoint[model_key][key]
        else:
            new_model_state[key] = temp_checkpoint[model_key][key]

    temp_netG.load_state_dict(new_model_state)

    if world_rank==0: print("Loaded model checkpoint [✓]")
    
    if world_rank==0: 
        print(f"2nd Model Featmult:")
        for i, param in enumerate(temp_netG.noise_stream.featmult):
            print(f"Parameter {i} size : {param.size()}")
            print(f"Parameter {i} mean : {torch.mean(param)}")

##########

if args.output == 'all' or args.output == 'flux':
    flux_params = YParams(os.path.abspath(args.yaml_config), args.flux_config)
    flux_params.data_size = in_size
    flux_exp_dir = os.path.join('/global/cfs/cdirs/m3900/cjacobus/expdir/expts/', args.flux_config)
    flux_params.experiment_dir = os.path.abspath(flux_exp_dir)
    flux_params.checkpoint_path = os.path.join(flux_exp_dir, 'checkpoints/ckpt.tar')

    if  world_rank==0:
        print("3rd Checkpoint Path: {}".format(flux_params.checkpoint_path))
        print("Initializing 3rd  model...")
        print("3rd Config: {}, {}".format(args.yaml_config, args.flux_config))

    flux_netG = networks.define_G(flux_params).to(device)

    if world_rank==0:
        print("Initialized model [✓]")
        print("Loading Checkpoint: {}".format(flux_params.checkpoint_path))

    flux_checkpoint = torch.load(flux_params.checkpoint_path, map_location=' :%d'%local_rank)

    new_model_state = OrderedDict()
    model_key = 'model_state_G' if 'model_state_G' in flux_checkpoint else 'state_dict'
    for key in flux_checkpoint[model_key].keys():
        if 'module.' in key: # model was stored using ddp which prepends module
            name = str(key[7:])
            new_model_state[name] = flux_checkpoint[model_key][key]
        else:
            new_model_state[key] = flux_checkpoint[model_key][key]

    flux_netG.load_state_dict(new_model_state)

    if world_rank==0: print("Loaded model checkpoint [✓]")

#exp_dir = os.path.join('/pscratch/sd/c/cjacobus/tsit/expts/', args.config)

     
if not args.dummy: 
    if args.make_copy:
        save_path = args.temp_path
        if  world_rank==0:
            print("Making Template Copy...")
            shutil.copy(args.scaffold, args.temp_path)   
            print("Made Copy")
        else:
            time.sleep(60)
        file_exists = exists(save_path)
    else:
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
               
        if False:
            del hf["derived_fields"]
            #del hf["derived_fields/tau_red"]
            #del hf["derived_fields/tau_real"]
            #del hf["derived_fields/HI_number_density"]
            
        if True:
            flux_red = hf.create_dataset("derived_fields/flux_red", (full_dim,full_dim,full_dim), dtype='<f4')
            flux_red.attrs['units'] = "(none)"
            
            flux_real = hf.create_dataset("derived_fields/flux_real", (full_dim,full_dim,full_dim), dtype='<f4')
            flux_real.attrs['units'] = "(none)"
            
            #HI = hf.create_dataset("derived_fields/HI_number_density", (full_dim,full_dim,full_dim), dtype='<f4')
            #HI.attrs['units'] = "cm**-3"
        
        print("Rank {} initialized file".format(str(world_rank).zfill(2)))
        
        if False and world_rank==0:
            hf['derived_fields']['tau_red'][:,:,:] = np.zeros((full_dim,full_dim,full_dim))
            hf['derived_fields']['tau_real'][:,:,:] = np.zeros((full_dim,full_dim,full_dim))
            hf['derived_fields']['HI_number_density'][:,:,:] = np.zeros((full_dim,full_dim,full_dim))
            print("Rank {} overwrote derived fields".format(str(world_rank).zfill(2)))
        
        if  world_rank==0: 
            print("Beginning...")
        
        f = h5py.File(args.datapath, 'r')
        rf = h5py.File(args.realpath, 'r')
        
        if  world_rank==0: print("Input data path: {}".format(args.datapath))
        full_dim = f['coarse'][0,0,0,:].shape[0]
        if  world_rank==0: print("Dimension: {}".format(full_dim))
        slices=int(full_dim/size)
        
        #del hf["chunk_complete"]
        hf.create_dataset("chunk_complete", (slices,slices,slices), dtype='<f4')
    
        if  world_rank==0:
            print("Dimension: {}".format(full_dim))
            print("Trimmed chunk size: {}".format(size))
            print("Trim: {}".format(trim))
            print("Slices: {}".format(slices))
            print("Sending {} chunks individually...".format(slices**3))
        
        
        time.sleep(world_rank*0.1)
        
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

                        if not hf['chunk_complete'][x,y,z] == 2:
                            time.sleep(world_rank*0.001)

                            if not hf['chunk_complete'][x,y,z] == 2:
                                hf['chunk_complete'][x,y,z] = 2
                                start = time.perf_counter()

                                x_plus = None if (x2_edge == 0) else -x2_edge
                                y_plus = None if (y2_edge == 0) else -y2_edge
                                z_plus = None if (z2_edge == 0) else -z2_edge

                                sliced_in = f['coarse'][:, x1-x1_edge:x2+x2_edge, y1-y1_edge:y2+y2_edge, z1-z1_edge:z2+z2_edge].astype(dtype)
                                sliced_in_t = np.expand_dims(sliced_in, axis=0) # add batch dim
                                sliced_in_t = torch.from_numpy(sliced_in_t).to(device)
                                
                                sliced_in_real = rf['fine'][:, x1-x1_edge:x2+x2_edge, y1-y1_edge:y2+y2_edge, z1-z1_edge:z2+z2_edge].astype(dtype)
                                sliced_in_real = np.expand_dims(sliced_in_real, axis=0) # add batch dim
                                sliced_in_real = torch.from_numpy(sliced_in_real).to(device)
                                
                                stop = time.perf_counter()
                                print("Rank {} received chunk [{},{},{}],   input shape: {},   took {}s".format(str(world_rank).zfill(2),x,y,z,sliced_in.shape,np.around(stop-start, decimals=3)))
                                
                                start = time.perf_counter()
                                with torch.no_grad():
                                    #chunk = model(sliced_in)
                                    if not args.output == 'flux':
                                        rho_chunk = rho_netG(sliced_in_t, sliced_in_real, z=None)
                                        print("Rank {} inferred DENS chunk [{},{},{}],   pred shape:  {},   took {}s".format(str(world_rank).zfill(2),x,y,z,rho_chunk.shape,np.around(stop-start, decimals=3)))
                                    
                                    if not args.output == 'dens' and not args.output == 'flux':
                                        temp_chunk = temp_netG(sliced_in_t, sliced_in_real, z=None)
                                        print("Rank {} inferred TEMP chunk [{},{},{}],   pred shape:  {},   took {}s".format(str(world_rank).zfill(2),x,y,z,temp_chunk.shape,np.around(stop-start, decimals=3)))
                                    
                                    if args.output == 'all' or args.output == 'flux':
                                        flux_chunk = flux_netG(sliced_in_t, sliced_in_real, z=None)
                                        print("Rank {} inferred FLUX chunk [{},{},{}],   pred shape:  {},   took {}s".format(str(world_rank).zfill(2),x,y,z,flux_chunk.shape,np.around(stop-start, decimals=3)))

                                
                                stop = time.perf_counter()
                                
                                #if not args.output == 'flux': print("Rank {} inferred chunk [{},{},{}],   pred shape:  {},   took {}s".format(str(world_rank).zfill(2),x,y,z,rho_chunk.shape,np.around(stop-start, decimals=3)))
                                start = time.perf_counter()
                                
                                if not args.output == 'flux':
                                    rho_chunk = rho_chunk.cpu()
                                    rho_chunk = rho_chunk.numpy()
                                
                                if not args.output == 'dens' and not args.output == 'flux':
                                    temp_chunk = temp_chunk.cpu()
                                    temp_chunk = temp_chunk.numpy()
                                    
                                if args.output == 'all' or args.output == 'flux':
                                    flux_chunk = flux_chunk.cpu()
                                    flux_chunk = flux_chunk.numpy()
                                #chunk = np.minimum(chunk,1)
                                #chunk = np.maximum(chunk,-1)
                                stop = time.perf_counter()
                                #print("Rank {} wrote    chunk [{},{},{}],   took {}s".format(str(world_rank).zfill(2) ,x,y,z, np.around(stop-start, decimals=3)))
                                
                                # un-normalize the NN output and write to file
                                #start = time.perf_counter()
                                
                                if args.native:
                                    #######
                                    if args.output == 'dens':
                                        out_dens = np.exp(14.*rho_chunk[0, 0, x1_edge:x_plus, y1_edge:y_plus, z1_edge:z_plus])
                                        out_dens = np.where( np.logical_and(out_dens > 0.05, out_dens < 1e4), out_dens, np.exp(14.*sliced_in[0,x1_edge:x_plus, y1_edge:y_plus, z1_edge:z_plus]))
                                        
                                        #out_dens = gaussian_filter(out_dens, sigma=1/8, mode='reflect', truncate=24.0)
                                        
                                        #out_dens = np.where(out_dens > 0, out_dens, 0.001)
                                        hf['native_fields']['baryon_density'][x1:x2,y1:y2,z1:z2] = out_dens
                                      
                                    
                                    #######
                                    elif args.output == 'temp':
                                        hf['native_fields']['temperature'][x1:x2,y1:y2,z1:z2] = np.exp(8.*(chunk[0, 0, x1_edge:x_plus, y1_edge:y_plus, z1_edge:z_plus] + 1.5))
                                     
                                    
                                    
                                    
                                    ####### 
                                    elif args.output == 'dual': 
                                        out_dens = np.maximum(np.minimum(np.exp(14.*rho_chunk[0, 0, x1_edge:x_plus, y1_edge:y_plus, z1_edge:z_plus]), 1e6),3e-3)
                                        hf['native_fields']['baryon_density'][x1:x2,y1:y2,z1:z2] = out_dens
                                        
                                        out_temp = np.maximum(np.minimum(np.exp(8.*(temp_chunk[0, 1, x1_edge:x_plus, y1_edge:y_plus, z1_edge:z_plus] + 1.5)), 1e8),1e2)
                                        hf['native_fields']['temperature'][x1:x2,y1:y2,z1:z2] = out_temp
                                     
                                    
                                    
                                    #######
                                    elif args.output == 'dub' or args.output == 'all':
                                        
                                        out_dens = np.maximum(np.minimum(np.exp(14.*rho_chunk[0, 0, x1_edge:x_plus, y1_edge:y_plus, z1_edge:z_plus]), 1e6),3e-3)
                                        
                                        in_dens = np.exp(14.*sliced_in[0, x1_edge:x_plus, y1_edge:y_plus, z1_edge:z_plus])
                                        in_mean = np.mean(in_dens)
                                        
                                        if args.mask:
                                            dens_mask =  np.where( np.logical_and(out_dens > 4e-2, out_dens < 1e4), 0.0, 5.0)
                                            #dens_mask =  np.where( np.logical_and(out_dens > 1e-2, out_dens < 1e4), 0.0, 5.0)
                                            dens_mask = gaussian_filter(dens_mask, sigma=0.5, mode='reflect', truncate=8.0)
                                            dens_mask = np.maximum(np.minimum( dens_mask, 1.0), 0.0)

                                            out_dens = np.add( np.multiply(out_dens, (1-dens_mask) ), np.multiply( in_dens, (dens_mask) ) )
                                            
                                        if args.blur and False: out_dens = gaussian_filter(out_dens, sigma=0.5, mode='reflect', truncate=16.0)
                                        
                                        if args.rescale:
                                            out_dens = out_dens * 1.06 #in_mean / out_mean
                                            #out_mean = np.mean(out_dens)
                                            #out_dens = out_dens * np.sqrt(in_mean / out_mean)
                                        
                                        
                                        #out_dens = np.where( np.logical_and(out_dens > 0.05, out_dens < 1e4), out_dens, np.exp(14.*sliced_in[0, x1_edge:x_plus, y1_edge:y_plus, z1_edge:z_plus]))
                                        hf['native_fields']['baryon_density'][x1:x2,y1:y2,z1:z2] = out_dens
                                        
                                        stop = time.perf_counter()
                                        print("Rank {} wrote   DENS chunk [{},{},{}],   took {}s".format(str(world_rank).zfill(2) ,x,y,z, np.around(stop-start, decimals=3)))
                                        
                                        out_temp = np.maximum(np.minimum(np.exp(8.*(temp_chunk[0, 0, x1_edge:x_plus, y1_edge:y_plus, z1_edge:z_plus] + 1.5)), 1e8),1e2)
                                        
                                        in_temp = np.exp(8.*(sliced_in[4, x1_edge:x_plus, y1_edge:y_plus, z1_edge:z_plus]+ 1.5))
                                        in_mean = np.mean(in_temp)
                                        
                                        if args.mask:
                                            #temp_mask =  np.where( np.logical_and(out_temp > 5e3, out_temp < 1e6), 0.0, 3.0*(np.log(out_temp)-1e5) )
                                            #temp_mask =  np.where(out_temp > 3e3, 0.0, 3.0)
                                            temp_mask =  np.where(out_temp > 2e3, 0.0, 3.0)
                                            temp_mask =  np.where(out_temp < 2e6, temp_mask, 0.4*(np.log(out_temp)-np.log(2e6)) )
                                            #temp_mask =  np.where(out_temp < 1e7, temp_mask, 0.4*(np.log(out_temp)-np.log(1e7)) )
                                            temp_mask = gaussian_filter(temp_mask, sigma=0.5, mode='reflect', truncate=8.0)
                                            temp_mask = np.maximum(np.minimum( temp_mask, 1.0), 0.0)
                                            out_temp = np.add( np.multiply(out_temp, (1-temp_mask) ), np.multiply( in_temp , (temp_mask) ) )
                                        if args.blur: out_temp = gaussian_filter(out_temp, sigma=1, mode='reflect', truncate=8.0)
                                        
                                        if args.rescale:
                                            out_mean = np.mean(out_temp)
                                            out_temp = out_temp * 0.97 #* 0.88 #* in_mean / out_mean 
                                        
                                        #out_temp = np.where( np.logical_and(out_temp > 2e3, out_temp < 1e7), out_temp, np.exp(8.*(sliced_in[4, x1_edge:x_plus, y1_edge:y_plus, z1_edge:z_plus] + 1.5)) )
                                        
                                        hf['native_fields']['temperature'][x1:x2,y1:y2,z1:z2] = out_temp
                                        stop = time.perf_counter()
                                        print("Rank {} wrote   TEMP chunk [{},{},{}],   took {}s".format(str(world_rank).zfill(2) ,x,y,z, np.around(stop-start, decimals=3)))
                                        
                                        if args.output == 'all':
                                            if args.flux:
                                                out_flux = np.maximum(np.minimum(flux_chunk[0, 0, x1_edge:x_plus, y1_edge:y_plus, z1_edge:z_plus], 1),1e-100)

                                                hf['derived_fields']['tau_red'][x1:x2,y1:y2,z1:z2] = -np.log(out_flux)
                                                hf['derived_fields']['flux_red'][x1:x2,y1:y2,z1:z2] = out_flux
                                        
                                    elif args.output == 'flux':
                                        if args.flux:
                                                out_flux = np.maximum(np.minimum(flux_chunk[0, 0, x1_edge:x_plus, y1_edge:y_plus, z1_edge:z_plus], 1),1e-100)

                                                hf['derived_fields']['tau_red'][x1:x2,y1:y2,z1:z2] = -np.log(out_flux)
                                                hf['derived_fields']['flux_red'][x1:x2,y1:y2,z1:z2] = out_flux
                                        
                                   
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
                                    
                                #stop = time.perf_counter()
                                #print("Rank {} wrote    chunk [{},{},{}],   took {}s".format(str(world_rank).zfill(2) ,x,y,z, np.around(stop-start, decimals=3)))
                                #print("Chunk [{},{},{}] output shape: {}".format(x,y,z,chunk.shape))

                        
    
        hf.close()
    
maxRSS = resource.getrusage(resource.RUSAGE_SELF)[2]

if  world_rank==0:
    print("MaxRSS: {} [GB]".format(maxRSS/1e6))

    print('DONE')