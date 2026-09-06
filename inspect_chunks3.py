import torch
import torchaudio
import soundfile as sf
import os
import matplotlib.pyplot as plt
import numpy as np

def get_audio(path):
    audio, sr = sf.read(path, always_2d=True)
    y = torch.from_numpy(audio.T.astype('float32')).mean(dim=0, keepdim=True)
    if sr != 44100:
        y = torchaudio.functional.resample(y, sr, 44100)
    return y

y_demucs = get_audio('output/stems/htdemucs/mp3uyU0HAQI/drums.wav')

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=44100, n_fft=2048, hop_length=441, n_mels=229, f_min=30.0, f_max=44100 / 2.0
)
amp_to_db = torchaudio.transforms.AmplitudeToDB()

# Global raw power spectrogram
mel_raw = mel_transform(y_demucs) # [1, 229, T]

# Chunked AmplitudeToDB
segment_frames = 300
mel_db = torch.zeros_like(mel_raw)

for i in range(0, mel_raw.shape[-1], segment_frames):
    end_idx = min(i + segment_frames, mel_raw.shape[-1])
    chunk = mel_raw[..., i:end_idx]
    mel_db[..., i:end_idx] = amp_to_db(chunk)

print(f"Chunked Demucs DB - min: {mel_db.min():.2f}, max: {mel_db.max():.2f}, mean: {mel_db.mean():.2f}")

