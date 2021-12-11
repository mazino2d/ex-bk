import random as rd

import numpy as np
from pydub import AudioSegment


def read_mp3(path: str) -> np.ndarray:
    audio = AudioSegment.from_mp3(path)
    return np.array(audio.get_array_of_samples())


def pick_audio(data: np.ndarray, length: int) -> np.ndarray:
    size = data.shape[0] - length
    if size < 0:
        size = data.shape[0] // 2
    start = rd.randint(0, size)
    output = data[start: start + length]
    len_out = output.shape[0]
    if len_out < length:
        output = np.pad(output, (0, length - len_out), "constant")
    return output


if __name__ == "__main__":
    filename = "/home/khoidd/project/ex-bk/data/train/train/song/0000.mp3"
    data = read_mp3(filename)
    print(type(data))
    print(data.shape)
    seg = pick_audio(np.array([1, 2, 3, 4, 5, 6]), 2)
    print(type(seg))
    print(seg)
    print(seg.shape)
    seg = pick_audio(np.array([1, 2, 3]), 10)
    print(type(seg))
    print(seg)
    print(seg.shape)
