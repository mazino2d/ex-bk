import random as rd

import tensorflow as tf
import tensorflow_io as tfio


def read_aac(path: str) -> tf.Tensor:
    audio = tfio.audio.decode_aac(tf.io.read_file(path))
    return audio


def read_flac(path: str) -> tf.Tensor:
    audio = tfio.audio.decode_flac(tf.io.read_file(path))
    return audio


def read_mp3(path: str) -> tf.Tensor:
    audio = tfio.audio.decode_mp3(tf.io.read_file(path))
    return audio


def read_wav(path: str) -> tf.Tensor:
    audio = tfio.audio.decode_wav(tf.io.read_file(path))
    return audio


def pick_audio(data: tf.Tensor, length: int) -> tf.Tensor:
    size = data.shape[0] - length
    if size < 0:
        size = data.shape[0] // 2
    start = rd.randint(0, size)
    output = data[start: start + length, :]

    len_out = output.shape[0]
    print(start)
    print(len_out)
    if len_out < length:
        paddings = tf.constant([[0, length - len_out], [0, 0]])
        output = tf.pad(output, paddings, "CONSTANT")
    return output
