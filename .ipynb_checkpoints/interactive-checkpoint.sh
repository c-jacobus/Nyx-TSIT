#!/bin/bash 

ROOT_DIR=$SCRATCH/tsit
mkdir -p ${ROOT_DIR}

LOGDIR=${SCRATCH}/ML_Hydro_train/logs
mkdir -p ${LOGDIR}

export HDF5_USE_FILE_LOCKING=FALSE
args="${@}"

export NCCL_NET_GDR_LEVEL=PHB
export MASTER_ADDR=$(hostname)


set -x
srun -u --module=gpu,nccl-2.15 --ntasks-per-node 4 --cpus-per-task 32 --gpus-per-node 4 shifter -V ${LOGDIR}:/logs --image=nersc/pytorch:ngc-22.03-v0 --env PYTHONUSERBASE=$HOME/.local/perlmutter/nersc-pytorch-ngc-22.03-v0 \
    bash -c "
    source export_DDP_vars.sh
    python train.py --root_dir=${ROOT_DIR} ${args}
    "
