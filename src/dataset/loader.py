
import pandas as pd
import tqdm
from src.util import audio


def load(path_meta="data/train/train_meta.csv"):
    train_meta = pd.read_csv(path_meta)
    ls_data = []
    for _, row in tqdm.tqdm(list(train_meta.iterrows())):
        song = audio.read_mp3("data/" + row["song_path"])
        hum = audio.read_mp3("data/" + row["hum_path"])
        ls_data.append([song, hum])
    df = pd.DataFrame(ls_data, columns=["song", "hum"])
    return df
