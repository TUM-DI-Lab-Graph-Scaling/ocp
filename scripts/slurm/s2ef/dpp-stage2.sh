#!/bin/bash
#SBATCH --mincpus=128
#SBATCH --mem=500gb
#SBATCH --gres=gpu:8
#SBATCH --time=14:00:00
#SBATCH --nodelist=lundquist
#SBATCH --job-name="s2ef-dpp"

conda activate ocp-models

export PATH=/usr/local/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.1/lib64:/usr/local/extras/CUPTI/lib64:/usr/local/lib:$LD_LIBRARY_PATH
export CUDADIR=/usr/local/cuda-11.1
export CUDA_HOME=/usr/local/cuda-11.1
export NCCL_P2P_LEVEL=PIX

srun python -u -m torch.distributed.launch --nproc_per_node=8 main.py \
  --distributed --num-gpus 8 --mode train --config-yml configs/s2ef/200k/dimenet_plus_plus/dpp.yml \
  --deepspeed-config configs/s2ef/200k/dimenet_plus_plus/ds_config_stage2.json
