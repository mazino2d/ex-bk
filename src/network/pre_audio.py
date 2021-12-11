import tensorflow.keras.layers as La
from kapre import STFT, ApplyFilterbank, Magnitude, MagnitudeToDecibel


def melspectrogram(
    input_shape=None,
    n_fft=2048,
    win_length=None,
    hop_length=None,
    window_name=None,
    pad_begin=False,
    pad_end=False,
    sample_rate=22050,
    n_mels=128,
    mel_f_min=0.0,
    mel_f_max=None,
    mel_htk=False,
    mel_norm='slaney',
    db_amin=1e-5,
    db_ref_value=1.0,
    db_dynamic_range=80.0,
    input_data_format='channels_last',
    output_data_format='channels_last',
    postfix='1',
):
    inpt = La.Input(shape=input_shape, name="input_" + postfix)

    oupt = STFT(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        window_name=window_name,
        pad_begin=pad_begin,
        pad_end=pad_end,
        input_data_format=input_data_format,
        output_data_format=output_data_format,
        name="stft_" + postfix
    )(inpt)

    oupt = Magnitude(
        name="mag_" + postfix
    )(oupt)

    oupt = ApplyFilterbank(
        type='mel',
        filterbank_kwargs={
            'sample_rate': sample_rate,
            'n_freq': n_fft // 2 + 1,
            'n_mels': n_mels,
            'f_min': mel_f_min,
            'f_max': mel_f_max,
            'htk': mel_htk,
            'norm': mel_norm,
        },
        data_format=output_data_format,
        name="mel_" + postfix
    )(oupt)

    oupt = MagnitudeToDecibel(
        ref_value=db_ref_value,
        amin=db_amin,
        dynamic_range=db_dynamic_range,
        name="mag2dec_" + postfix
    )(oupt)

    return inpt, oupt
