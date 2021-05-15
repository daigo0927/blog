#!/bin/sh

CUDA_VISIBLE_DEVICES=0,1,2,3 python train_tf_multiworker_mirroredstrategy.py -wi 0 &> job_0.log
CUDA_VISIBLE_DEVICES=4,5,6,7 python train_tf_multiworker_mirroredstrategy.py -wi 1
unset CUDA_VISIBLE_DEVICES
