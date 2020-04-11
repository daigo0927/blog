'''
Training script.
Reference: https://www.tensorflow.org/tensorboard/get_started
'''

import os
import time
import argparse
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import optimizers, losses, metrics

from models import CNN, ReparamCNN, FlipOutCNN


def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def train(model_type, epochs, batch_size, logdir):
    if model_type == 'normal':
        model = CNN(num_classes=10)
    elif model_type == 'reparam':
        model = ReparamCNN(num_classes=10)
    else:
        model = FlipOutCNN(num_classes=10)

    ds_train, ds_test = tfds.load('cifar10',
                                  split=['train', 'test'],
                                  as_supervised=True)
    ds_train = ds_train.shuffle(50000)\
      .batch(batch_size)\
      .map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
      .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    ds_test = ds_test.batch(batch_size)\
      .map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
      .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    loss_fn = losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = optimizers.Adam()

    train_loss = metrics.Mean('train_loss', dtype=tf.float32)
    train_acc = metrics.SparseCategoricalAccuracy('train_acc')
    test_loss = metrics.Mean('test_loss', dtype=tf.float32)
    test_acc = metrics.SparseCategoricalAccuracy('test_acc')

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            logits = model(images, training=True)
            loss = loss_fn(labels, logits)
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        train_loss(loss)
        train_acc(labels, logits)

    @tf.function
    def test_step(images, labels):
        logits = model(images)
        loss = loss_fn(labels, logits)
        test_loss(loss)
        test_acc(labels, logits)

    train_logdir = logdir + '/train'
    test_logdir = logdir + '/test'
    train_sw = tf.summary.create_file_writer(train_logdir)
    test_sw = tf.summary.create_file_writer(test_logdir)

    for epoch in range(epochs):
        for images, labels in ds_train:
            train_step(images, labels)
        with train_sw.as_default():
            tf.summary.scaler('loss', train_loss.result(), step=epoch)
            tf.summary.scaler('accuracy', train_acc.result(), step=epoch)

        for images, labels in ds_test:
            test_step(images, labels)
        with test_sw.as_default():
            tf.summary.scaler('loss', test_loss.result(), step=epoch)
            tf.summary.scaler('accuracy', test_acc.result(), step=epoch)

        print('\nEpoch: {}, loss: {}, acc: {}, test loss: {}, test acc: {}'\
              .format(epoch+1, train_loss.result(), train_acc.result(),
                      test_loss.result(), test_acc.result()))
        train_loss.reset_states()
        train_acc.reset_states()
        test_loss.reset_states()
        test_acc.reset_states()

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
