#!/bin/sh

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nnodes=2 --node_rank=0 --nproc_per_node=4 train_torch_ddp.py ../../datasets/stanford_dogs > job_torch_ddp_0.log &
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nnodes=2 --node_rank=1 --nproc_per_node=4 train_torch_ddp.py ../../datasets/stanford_dogs
unset CUDA_VISIBLE_DEVICES
