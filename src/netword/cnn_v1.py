from typing import Tuple

import tensorflow as tf
import tensorflow.keras.backend as Ba
import tensorflow.keras.layers as La

from . import preaudio


def build(
        duration: int = 5,
        sample_rate: int = 22050,
        pool_list=[(3, 4), (2, 2), (2, 2), (2, 2), (2, 2), (2, 2), (2, 1)]
) -> Tuple[La.Layer, La.Layer, La.Layer]:

    input_shape = (duration * sample_rate, 1)
    inpt1, branch1 = preaudio.melspectrogram(input_shape, postfix="1")
    inpt2, branch2 = preaudio.melspectrogram(input_shape, postfix="2")
    def stack_func(x): return Ba.stack([x[0], x[1]], axis=1)
    oupt = La.Lambda(stack_func, name="stack",)([branch1, branch2])

    cnn_layers = []
    for blk_idx, blk_shape in enumerate(pool_list):
        conv = La.Conv2D(
            filters=8,
            kernel_size=(2, 2),
            padding='same',
            name=f'conv{blk_idx}',
        )
        bn = La.BatchNormalization(
            axis=3,
            name=f'bn{blk_idx}',
        )
        relu = La.Activation(
            activation='relu',
            name=f'relu{blk_idx}',
        )
        pool = La.MaxPooling2D(
            pool_size=blk_shape,
            name=f'pool{blk_idx}',
        )
        dropout = La.Dropout(
            rate=0.25,
            name=f'dropout{blk_idx}',
        )
        cnn_layers.extend([conv, bn, relu, pool, dropout])

    for layer in cnn_layers:
        oupt = La.TimeDistributed(layer, name=layer.name)(oupt)

    def dot_product(data: tf.Tensor):
        embed0: tf.Tensor = data[:, 0, :, :, :]
        embed1: tf.Tensor = data[:, 1, :, :, :]
        return Ba.sum(embed0 * embed1, axis=(1, 2, 3))
    oupt = La.Lambda(dot_product, name="classify")(oupt)

    return inpt1, inpt2, oupt
