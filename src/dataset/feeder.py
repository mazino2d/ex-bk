import tensorflow as tf


def build(batch_size: int = 8):

    while True:
        yield (tf.random.uniform((batch_size, 110250, 1)), tf.random.uniform((batch_size, 110250, 1))), tf.random.uniform((batch_size,))
