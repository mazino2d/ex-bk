
import pandas as pd
import tensorflow as tf
from src.util import audio


def load(path_meta="data/train/train/train_meta.csv"):
    train_meta = pd.read_csv(path_meta)
    ls_data = []
    for _, row in train_meta.iterrows():
        song = audio.read_mp3("data/train/" + row["song_path"])
        hum = audio.read_mp3("data/train/" + row["hum_path"])
        ls_data.append([
            tf.expand_dims(song, axis=0),
            tf.expand_dims(hum, axis=0)]
        )
    df = pd.DataFrame(ls_data, columns=["song", "hum"])
    return df
