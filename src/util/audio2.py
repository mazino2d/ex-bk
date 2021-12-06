import tensorflow as tf
import librosa
import numpy as np


def read_mp3(path: str):
    # audio = librosa.load(path)
    song, sr = np.array(librosa.load(path))
    print(type(song))
    print(song.shape)
    print(np.sum(song))
    print(song.ndim)
    return song


def read_wav(path: str) -> tf.Tensor:
    audio = librosa.load(path)
    return audio

if __name__ == "__main__":
    read_mp3("data/train/hum/0010.mp3")
