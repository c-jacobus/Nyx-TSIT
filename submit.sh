#!/bin/bash -l
#SBATCH --time=06:00:00
#SBATCH -C gpu
#SBATCH --account=nyx_g
#SBATCH -q regular
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --image=nersc/pytorch:ngc-22.02-v0

ROOT_DIR=/global/cfs/cdirs/m3900/cjacobus/expdir
mkdir -p ${ROOT_DIR}

LOGDIR=/global/cfs/cdirs/m3900/cjacobus/logdir
mkdir -p ${LOGDIR}

export HDF5_USE_FILE_LOCKING=FALSE
args="${@}"

export FI_MR_CACHE_MONITOR=userfaultfd
export NCCL_NET_GDR_LEVEL=PHB
export MASTER_ADDR=$(hostname)

set -x
srun -u shifter --module=gpu,nccl-2.15 --image=nersc/pytorch:ngc-22.03-v0 --env PYTHONUSERBASE=$HOME/.local/perlmutter/nersc-pytorch-ngc-22.03-v0 \
    bash -c "
    source export_DDP_vars.sh
    python train.py --root_dir=${ROOT_DIR} ${args}
    "