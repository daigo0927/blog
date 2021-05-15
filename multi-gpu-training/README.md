# Multi-GPU training with TensorFlow/PyTorch

## TensorFlow

- [Training with distribute strategy](https://www.tensorflow.org/guide/distributed_training?hl=ja)
- [Using GPUs](https://www.tensorflow.org/guide/gpu?hl=ja)
- [Using TPUs](https://www.tensorflow.org/guide/tpu?hl=ja)
- [Examples and Tutorials](https://www.tensorflow.org/guide/distributed_training?hl=ja#examples_and_tutorials)
- [MultiWorker with Keras](https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras)

## PyTorch

- [Distributed overview](https://pytorch.org/tutorials/beginner/dist_overview.html)
- [Distributed Data Parallel](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)

# Sample scripts

## TensorFlow

- `train_tf_mirroredstrategy.py`: Fast

## PyTorch

- `train_torch_dp.py`: Fast, but data loading seems to be bottleneck
- `train_torch_ddp.py`: WIP
