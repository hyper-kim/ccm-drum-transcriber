import torch
import torchaudio
import soundfile as sf
import os
import matplotlib.pyplot as plt

def get_spec_raw(path):
    audio_np, sr = sf.read(path, always_2d=True)
    y = torch.from_numpy(audio_np.T.astype("float32")).mean(dim=0, keepdim=True)
    if sr != 44100:
        y = torchaudio.functional.resample(y, sr, 44100)
    
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=44100, n_fft=2048, hop_length=441, n_mels=229, f_min=30.0, f_max=44100 / 2.0
    )
    
    # Just return raw power mel spectrogram
    return mel_transform(y).squeeze(0)

# 1. Groove Dataset (Training)
import pandas as pd
df = pd.read_csv('data/groove/info.csv')
groove_path = os.path.join('data/groove', df.iloc[0]['audio_filename'])
raw_groove = get_spec_raw(groove_path)

# 2. HTDemucs (Inference)
raw_demucs = get_spec_raw('output/stems/htdemucs/mp3uyU0HAQI/drums.wav')

print(f"Raw Groove Power - min: {raw_groove.min():.10f}, max: {raw_groove.max():.10f}")
print(f"Raw Demucs Power - min: {raw_demucs.min():.10f}, max: {raw_demucs.max():.10f}")

# Find exactly how dataset.py applies amplitude_to_db during training
amp_to_db = torchaudio.transforms.AmplitudeToDB()

db_groove = amp_to_db(raw_groove)
db_demucs = amp_to_db(raw_demucs)

print(f"\nDefault AmpToDB Groove - min: {db_groove.min():.2f}, max: {db_groove.max():.2f}")
print(f"Default AmpToDB Demucs - min: {db_demucs.min():.2f}, max: {db_demucs.max():.2f}")

