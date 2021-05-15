import os
import json
import argparse
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB2

SEED = 42
N_CLASSES = 120
IMAGE_SIZE = (128, 128)
AUTOTUNE = tf.data.experimental.AUTOTUNE


TF_CONFIG = {
    'cluster': {
        'worker': ['localhost:12345']
    },
    'task': {'type': 'worker', 'index': 0}
}
os.environ['TF_CONFIG'] = json.dumps(TF_CONFIG)


def build_strategy(n_gpus):
    config = json.loads(os.environ['TF_CONFIG'])
    task_info = config['task']
    print('Current machine task:', task_info)

    strategy = tf.distribute.MultiWorkerMirroredStrategy()    
    
    gpus = tf.config.list_physical_devices('GPU')
    print('Available GPUs:', gpus)
    tf.config.set_visible_devices(gpus[:n_gpus], 'GPU')
    logical_gpus = tf.config.list_logical_devices('GPU')
    print('Visible logical gpus:', logical_gpus)
    return strategy


def build_network(image_size, n_classes):
    image = layers.Input([*image_size, 3], dtype=tf.float32, name='input')

    effnet = EfficientNetB2(include_top=False,
                            weights='imagenet',
                            pooling='avg')
    feature = effnet(image)
    feature = layers.Dropout(0.5)(feature)
    logit = layers.Dense(n_classes)(feature)
    model = tf.keras.Model(inputs=image, outputs=logit)
    return model


def preprocess(image, label):
    image = tf.image.resize_with_pad(image, *IMAGE_SIZE)
    image = tf.cast(image, dtype=tf.float32)
    return image, label


def augment(image, label):
    image = tf.image.random_brightness(image, 0.3, SEED)
    image = tf.image.random_contrast(image, 0.5, 1.0, SEED)
    image = tf.image.random_flip_left_right(image, SEED)
    return image, label


def run(n_gpus, epochs, batch_size, learning_rate):
    strategy = build_strategy(n_gpus)

    ds_train, ds_val = tfds.load('stanford_dogs', split=['train', 'test'],
                                 as_supervised=True)

    ds_train = ds_train.map(preprocess, num_parallel_calls=AUTOTUNE)
    ds_train = ds_train.map(augment, num_parallel_calls=AUTOTUNE)
    ds_train = ds_train.shuffle(buffer_size=len(ds_train), seed=SEED, reshuffle_each_iteration=True)
    ds_train = ds_train.batch(batch_size).prefetch(buffer_size=batch_size)

    ds_val = ds_val.map(preprocess, num_parallel_calls=AUTOTUNE)
    ds_val = ds_val.batch(batch_size).prefetch(buffer_size=batch_size)

    with strategy.scope():
        model = build_network(IMAGE_SIZE, N_CLASSES)
        optimizer = tf.keras.optimizers.Adam(learning_rate)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metrics = tf.keras.metrics.SparseCategoricalAccuracy()

    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)

    model.fit(
        ds_train,
        epochs=epochs,
        callbacks=[tf.keras.callbacks.CSVLogger('training.log')],
        validation_data=ds_val,
        steps_per_epoch=len(ds_train),
        validation_steps=len(ds_val)
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TF multi-GPU training')
    parser.add_argument('-n', '--n-gpus', type=int, default=8,
                        help='Number of gpus to use, [8] default')
    parser.add_argument('-e', '--epochs', type=int, default=10,
                        help='Number of epochs, [10] default')
    parser.add_argument('-bs', '--batch-size', type=int, default=1024,
                        help='Batch size, [1024] default')
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.001,
                        help='Learning rate, [0.001] default')
    args = parser.parse_args()

    run(**vars(args))
