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

## TensorFlow

- `train_tf_mirroredstrategy.py`: Fast
- `train_tf_multiworker_mirroredstrategy.py`: Fast
  - `run_tf_mwms.sh`: Shell script for running multi-worker training

## PyTorch

- `train_torch_dp.py`: Fast, but data loading seems to be a bottleneck_features
- `train_torch_ddp.py`: WIP

