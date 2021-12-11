import os
import pickle

import tensorflow as tf
import tensorflow.keras as keras
from annoy import AnnoyIndex
from src.network import cnn_2s
from src.util import audio

if __name__ == "__main__":
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    # trained model
    inpt1, inpt2, oupt = cnn_2s.build()
    model_train = keras.Model(
        inputs=[inpt1, inpt2],
        outputs=oupt,
    )
    model_train.load_weights("tmp/weights.126.hdf5")
    model_train.summary()
    # deployed model
    inpt, oupt = cnn_2s.deploy()
    model_deploy = keras.Model(inputs=inpt, outputs=oupt)
    model_deploy.summary()

    for layer in model_deploy.layers:
        layer.set_weights(model_train.get_layer(layer.name).get_weights())

    path_full_song = "data/public_test/full_song"

    annoy_index = AnnoyIndex(8, 'dot')
    map_song_idx = {}
    for idx, filename in enumerate(os.listdir(path_full_song)):
        path = f"{path_full_song}/{filename}"
        print(path)
        song = audio.read_mp3(path)
        melody = audio.pick_audio(song, 22050*5)
        melody = tf.expand_dims(melody, axis=0)
        melody = tf.expand_dims(melody, axis=-1)
        output = tf.squeeze(model_deploy.predict(melody))
        id_song = int(filename.replace(".mp3", ''))
        map_song_idx[idx] = id_song
        annoy_index.add_item(idx, output)

    with open("tmp/map.pickle", "wb") as f:
        pickle.dump(map_song_idx, f)
    annoy_index.build(10)
    annoy_index.save('tmp/result.ann')

    path_hum = "data/public_test/hum"
    writer = open("tmp/result.csv", "w")
    for idx, filename in enumerate(os.listdir(path_hum)):
        path = f"{path_hum}/{filename}"
        song = audio.read_mp3(path)
        melody = audio.pick_audio(song, 22050*5)
        melody = tf.expand_dims(melody, axis=0)
        melody = tf.expand_dims(melody, axis=-1)
        output = tf.squeeze(model_deploy.predict(melody))
        predict = annoy_index.get_nns_by_vector(output, 10)
        predict = [str(map_song_idx[e]) for e in predict]
        writer.write(filename + "," + ",".join(predict) + "\n")
