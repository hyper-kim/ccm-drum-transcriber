import torch
import torchaudio
import soundfile as sf
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def get_audio(path):
    audio, sr = sf.read(path, always_2d=True)
    y = torch.from_numpy(audio.T.astype('float32')).mean(dim=0, keepdim=True)
    if sr != 44100:
        y = torchaudio.functional.resample(y, sr, 44100)
    return y

# 1. Groove Dataset (Training)
df = pd.read_csv('data/groove/info.csv')
groove_path = os.path.join('data/groove', df.iloc[0]['audio_filename'])
y_groove = get_audio(groove_path)

# 2. HTDemucs (Inference)
y_demucs = get_audio('output/stems/htdemucs/mp3uyU0HAQI/drums.wav')

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=44100, n_fft=2048, hop_length=441, n_mels=229, f_min=30.0, f_max=44100 / 2.0
)
amp_to_db = torchaudio.transforms.AmplitudeToDB()

# Take 3 seconds of Groove
y_g_chunk = y_groove[:, 44100*10:44100*13]
# Peka normalize
y_g_chunk = y_g_chunk / (y_g_chunk.abs().max() + 1e-8)
mel_g = amp_to_db(mel_transform(y_g_chunk)).squeeze(0).numpy()

# Take 3 seconds of Demucs
y_d_chunk = y_demucs[:, 44100*30:44100*33]
# Peak normalize
y_d_chunk = y_d_chunk / (y_d_chunk.abs().max() + 1e-8)
mel_d = amp_to_db(mel_transform(y_d_chunk)).squeeze(0).numpy()

print(f"Normalized Groove chunk - min: {mel_g.min():.2f}, max: {mel_g.max():.2f}, mean: {mel_g.mean():.2f}")
print(f"Normalized Demucs chunk - min: {mel_d.min():.2f}, max: {mel_d.max():.2f}, mean: {mel_d.mean():.2f}")

