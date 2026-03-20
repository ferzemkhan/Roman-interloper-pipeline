#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=desi
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --constraint=cpu
#SBATCH --qos=regular
#SBATCH --time=4:00:00
#SBATCH -J Roman_pipeline

#SBATCH --mail-type=ALL
#SBATCH --mail-user=yuyuwang@ohio.edu
#SBATCH -D /global/homes/y/yuyuwang/Roman/BAO_fitting_pipeline
#SBATCH -e /global/homes/y/yuyuwang/report/%x-%A.err
#SBATCH -o /global/homes/y/yuyuwang/report/%x-%A.out

module load conda
conda activate /pscratch/sd/y/yuyuwang/conda/cosmodesi-main-clone

export CUDA_VISIBLE_DEVICES=""

srun -n 1 --cpu-bind=cores python run_pipeline.py