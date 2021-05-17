# Multi-GPU training with TensorFlow/PyTorch

## TensorFlow

- [Training with distribute strategy](https://www.tensorflow.org/guide/distributed_training?hl=ja)
- [Using GPUs](https://www.tensorflow.org/guide/gpu?hl=ja)
- [Using TPUs](https://www.tensorflow.org/guide/tpu?hl=ja)
- [Examples and Tutorials](https://www.tensorflow.org/guide/distributed_training?hl=ja#examples_and_tutorials)
- [MultiWorker with Keras](https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras)

## PyTorch

- [Distributed overview](https://pytorch.org/tutorials/beginner/dist_overview.html)
- [Options for data-parallel training](https://pytorch.org/tutorials/beginner/dist_overview.html#data-parallel-training)
- [Distributed Data Parallel](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
  - [DDP materials](https://pytorch.org/tutorials/beginner/dist_overview.html#torch-nn-parallel-distributeddataparallel)
  - [Launching and configuring distributed data parallel applications](https://github.com/pytorch/examples/tree/master/distributed/ddp)  
  - [torch/distributed/launch.py](https://github.com/pytorch/pytorch/blob/master/torch/distributed/launch.py)


# Sample scripts

Scirpts are confirmed on a node with 8 A100 GPUs. For the multi-node experiment, I split the 8 GPUs into 2x4GPUs for simulating a multi-node environment.

## Environment

I have done the experiments with below environment. Different version of libraries may be acceptable but not tested.

- Machine settings
  - 8 NVIDIA A100 GPUs
  - python 3.8.7
  - cuda 11.0.3
  - cudnn 8.2.0
  - nccl 2.8.4-1
- Python libraries
  - tensorflow 2.4.0
  - torch 1.8.1+cu111
  - others listed in `requirements.txt`

## TensorFlow

- `train_tf_mirroredstrategy.py`: Basic data parallel training on a single node
- `train_tf_multiworker_mirroredstrategy.py`: Data parallel training with 2 nodes (i.e. workers)
  - `run_tf_mwms.sh`: Shell script for running multi-worker training

## PyTorch

- `train_torch_dp.py`: Basic data parallel training on a single node
- `train_torch_ddp.py`: Data parallel training with 2 nodes
  - `run_torch_ddp.sh`: Shell script for running DDP training

