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
