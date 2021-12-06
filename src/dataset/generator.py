import os
import random

from src.util import audio

import numpy as np
import pandas as pd
from tensorflow import keras

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, anotation_file, duration=5, sr=22050, batch_size=8, n_channels=1,
                 n_classes=2, shuffle=True):
        'Initialization'
        self.dim = (duration * sr, n_channels)
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.dataset=self.__load_data__(anotation_file)
        self.on_epoch_end()
        print(f"##### len : {self.__len_data__()}")

    def __load_data__(self, anotation_file, data_folder="./data"):
        data = pd.read_csv(anotation_file)
        list_hum_ids = list(data["hum_path"])
        list_song_ids = list(data["song_path"])

        dataset = []

        for i in range(len(list_song_ids)):
            try:
                path_hum = f"{data_folder}/{list_hum_ids[i]}"
                # print(f"path_hum : {path_hum}")
                src_hum = audio.read_mp3(path_hum)
                if src_hum.ndim > 1:
                    src_hum = np.mean(src_hum, axis=1)
                src_hum = np.expand_dims(src_hum, axis=1)
                path_song = f"{data_folder}/{list_song_ids[i]}"
                # print(f"path_song : {path_song}")
                src_song = audio.read_mp3(path_song)
                if src_song.ndim > 1:
                    src_song = np.mean(src_song, axis=1)
                src_song = np.expand_dims(src_song, axis=1)

                dataset.append((src_hum, src_song))
            except:
                continue
            
        return dataset

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.dataset) / self.batch_size))

    def __len_data__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        # indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        # list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(index)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.dataset))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, index):
        'Generates data containing batch_size samples' # X : (n_samples, n_pair, *dim)
        # Initialization
        X = np.empty((self.batch_size, 2, *self.dim))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        list_index2 = [index]
        choices = set(range(self.__len_data__())) - {index}
        list_index2.extend(random.sample(list(choices), self.batch_size - 1))
        for i in range(0, len(list_index2)):
            hum_audio = np.array(self.dataset[index][0][:self.dim[0],])
            if hum_audio.shape[0] < self.dim[0]:
                hum_audio = np.concatenate((hum_audio, np.zeros((self.dim[0] - hum_audio.shape[0], self.dim[1]), dtype=hum_audio.dtype)), axis=0)
            song_audio = np.array(self.dataset[list_index2[i-1]][1][:self.dim[0],])
            if song_audio.shape[0] < self.dim[0]:
                song_audio = np.concatenate((song_audio, np.zeros((self.dim[0] - song_audio.shape[0], self.dim[1]), dtype=song_audio.dtype)), axis=0)
            data = np.array((hum_audio, song_audio))
            X[i,] = data
            y[i] = 0

        return X, y


if __name__ == "__main__":
    # Parameters
    params = {'anotation_file': './data/train/train_meta.csv',
            'duration': 5, 
            'sr': 22050,
            'batch_size': 8,
            'n_channels': 1,
            'n_classes': 2,
            'shuffle': True}

    # training_generator = DataGenerator(**params)
    # training_generator[0]
    audio = audio.read_mp3("data/train/song/0005.mp3")
    print(audio.shape)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    audio = np.expand_dims(audio, axis=1)
    print(audio.dtype)
    print(np.zeros((3,1), dtype=audio.dtype))