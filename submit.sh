#!/bin/bash -l
#SBATCH --time=06:00:00
#SBATCH -C gpu
#SBATCH --account=m3900_g
#SBATCH -q regular
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --image=nersc/pytorch:ngc-22.02-v0

LOGDIR=${SCRATCH}/ML_Hydro_train/logs

ROOT_DIR=$SCRATCH/tsit
export HDF5_USE_FILE_LOCKING=FALSE
args="${@}"

export NCCL_NET_GDR_LEVEL=PHB
export MASTER_ADDR=$(hostname)

set -x
srun -u shifter -V ${LOGDIR}:/logs --image=nersc/pytorch:ngc-22.03-v0 --env PYTHONUSERBASE=$HOME/.local/perlmutter/nersc-pytorch-ngc-22.03-v0 \
    bash -c "
    source export_DDP_vars.sh
    python train.py --root_dir=${ROOT_DIR} ${args}
    "

#shifter --module gpu \
#    bash -c "
#    python train.py --root_dir=${ROOT_DIR} ${args}
#    "
