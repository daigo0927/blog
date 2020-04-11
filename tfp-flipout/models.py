''' 3 types of CNN (normal/reparameterized/flipout) '''

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers


def CNN(num_classes=10, **kwargs):
    model = tf.keras.Sequential([
        layers.Conv2D(64, 3, 2, 'same', activation=tf.nn.relu, **kwargs),
        layers.Conv2D(64, 3, 1, 'same', activation=tf.nn.relu, **kwargs),
        layers.Conv2D(128, 3, 2, 'same', activation=tf.nn.relu, **kwargs),
        layers.Conv2D(128, 3, 1, 'same', activation=tf.nn.relu, **kwargs),
        layers.Conv2D(num_classes, 3, 1, 'same', **kwargs),
        layers.GlobalAvgPool2D()
    ])
    return model


def ReparamCNN(num_classes, **kwargs):
    Conv2DReparam = tfp.layers.Convolution2DReparameterization

    model = tf.keras.Sequential([
        Conv2DReparam(64, 3, 2, 'same', activation=tf.nn.relu, **kwargs),
        Conv2DReparam(64, 3, 1, 'same', activation=tf.nn.relu, **kwargs),
        Conv2DReparam(128, 3, 2, 'same', activation=tf.nn.relu, **kwargs),
        Conv2DReparam(128, 3, 1, 'same', activation=tf.nn.relu, **kwargs),
        Conv2DReparam(num_classes, 3, 1, 'same'),
        layers.GlobalAvgPool2D()
    ])
    return model


def FlipOutCNN(num_classes, **kwargs):
    Conv2DFlipOut = tfp.layers.Convolution2DFlipout

    model = tf.keras.Sequential([
        Conv2DFlipOut(64, 3, 2, 'same', activation=tf.nn.relu, **kwargs),
        Conv2DFlipOut(64, 3, 1, 'same', activation=tf.nn.relu, **kwargs),
        Conv2DFlipOut(128, 3, 2, 'same', activation=tf.nn.relu, **kwargs),
        Conv2DFlipOut(128, 3, 1, 'same', activation=tf.nn.relu, **kwargs),
        Conv2DFlipOut(num_classes, 3, 1, 'same', **kwargs),
        layers.GlobalAvgPool2D()
    ])
    return model
