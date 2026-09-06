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

# Try standardization
mean_g, std_g = raw_groove.mean(), raw_groove.std()
mean_d, std_d = raw_demucs.mean(), raw_demucs.std()

norm_demucs = (raw_demucs - mean_d) / (std_d + 1e-6)
norm_demucs_matched = norm_demucs * std_g + mean_g

print(f"Matched Demucs min: {norm_demucs_matched.min():.2f}, max: {norm_demucs_matched.max():.2f}")

