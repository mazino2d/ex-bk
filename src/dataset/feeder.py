import time

import tensorflow as tf
from src.util import audio

from . import loader


def build(num_pos: int = 1, num_neg: int = 2, length=22050*5):
    df = loader.load("data/train/train/test_meta.csv")
    label = tf.constant([1]*num_pos + [0] * num_neg)
    while True:
        df_rand = df.sample(n=num_pos + num_neg).reset_index()
        one = audio.pick_audio(df_rand["hum"].iloc[0], length)
        sample1 = tf.stack([one]*(num_neg + num_pos))
        many = df_rand["song"]
        sample2 = many.map(lambda e: audio.pick_audio(e, length))
        sample2 = tf.stack(sample2)
        yield (sample1, sample2), label


if __name__ == "__main__":
    x = build(num_pos=1, )
    count = 100
    tm_start = time.time()
    for _ in range(count):
        data = next(x)
    tm_end = time.time()
    print(tm_end - tm_start)
