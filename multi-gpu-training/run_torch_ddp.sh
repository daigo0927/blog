#!/bin/sh

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr=localhost --master_port=12345 train_torch_ddp.py ../../datasets/stanford-dogs > job_torch_ddp_0.log &
CUDA_VISIBLE_DEVICES=4,5,6,7 python -m torch.distributed.launch --nproc_per_node=4 --nnodes=2 --node_rank=1 --master_addr=localhost --master_port=23456 train_torch_ddp.py ../../datasets/stanford-dogs
unset CUDA_VISIBLE_DEVICES
