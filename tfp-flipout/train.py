'''
Training script.
Reference: https://www.tensorflow.org/tensorboard/get_started
'''

import os
import time
import argparse
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import optimizers, losses
from tensorflow_probability import distributions as tfd

from models import CNN, ReparamCNN, FlipOutCNN


def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def train(model_type, epochs, batch_size, logdir):
    ds_train, ds_test = tfds.load('cifar10',
                                  split=['train', 'test'],
                                  as_supervised=True)
    num_samples = 50000
    num_samples_test = 10000
    ds_train = ds_train.shuffle(num_samples)\
      .batch(batch_size)\
      .map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
      .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    ds_test = ds_test.batch(batch_size)\
      .map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
      .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    kl_div_fn = (lambda q, p, _: tfd.kl_divergence(q, p) / num_samples)

    if model_type == 'normal':
        model = CNN(num_classes=10)
    elif model_type == 'reparam':
        model = ReparamCNN(num_classes=10, kernel_divergence_fn=kl_div_fn)
    else:
        model = FlipOutCNN(num_classes=10, kernel_divergence_fn=kl_div_fn)
    # Dummy inference, detailed in https://github.com/daigo0927/blog/issues/3
    # model.build(input_shape=[None, 32, 32, 3])
    # import numpy as np
    # _ = model(np.random.normal(size=(1, 32, 32, 3)))
    # import ipdb
    # ipdb.set_trace()

    optimizer = optimizers.Adam()
    loss_fn = losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = [
        tf.keras.metrics.SparseCategoricalCrossentropy(from_logits=True),
        tf.keras.metrics.SparseCategoricalAccuracy()
    ]
    model.compile(optimizer, loss=loss_fn, metrics=metrics)

    callbacks = [tf.keras.callbacks.TensorBoard(log_dir=logdir)]
    model.fit(ds_train,
              epochs=epochs,
              callbacks=callbacks,
              validation_data=ds_test,
              steps_per_epoch=num_samples // batch_size,
              validation_steps=num_samples_test)

    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CNN training')
    parser.add_argument('-m',
                        '--model_type',
                        choices=['normal', 'reparam', 'flipout'],
                        default='normal')
    parser.add_argument('-e', '--epochs', type=int, default=10)
    parser.add_argument('-b', '--batch_size', type=int, default=64)
    parser.add_argument('-d', '--logdir', type=str, default='logs')
    args = parser.parse_args()
    params = vars(args)

    time_str = time.strftime('%Y%m%d_%H%M%S')
    logdir = os.path.join(params['logdir'], time_str)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    params['logdir'] = logdir

    for k, v in params.items():
        print(f'{k}: {v}')

    train(**params)
