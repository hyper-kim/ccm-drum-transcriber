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
mel_g = amp_to_db(mel_transform(y_g_chunk)).squeeze(0).numpy()

# Take 3 seconds of Demucs
y_d_chunk = y_demucs[:, 44100*30:44100*33]
mel_d = amp_to_db(mel_transform(y_d_chunk)).squeeze(0).numpy()

print(f"Groove chunk - min: {mel_g.min():.2f}, max: {mel_g.max():.2f}, mean: {mel_g.mean():.2f}")
print(f"Demucs chunk - min: {mel_d.min():.2f}, max: {mel_d.max():.2f}, mean: {mel_d.mean():.2f}")

# Wait, if AmplitudeToDB is applied on a chunk, the max is always around 40-80 depending on the absolute peak.
# Wait! AmplitudeToDB by default scales relative to max(power). So the max DB value is always relative to the peak power.
# If Demucs has very low absolute peak power, its max DB value will be mathematically identical to Groove IF the relative noise floor is the same!
# Let's see the actual DB values.
