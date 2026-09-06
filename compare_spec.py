import torch
import torchaudio
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np

device = 'cpu'
mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=44100, n_fft=2048, hop_length=441, n_mels=229, f_min=30.0, f_max=44100 / 2.0
)
amp_to_db = torchaudio.transforms.AmplitudeToDB()

def get_spec(path):
    audio_np, sr = sf.read(path, always_2d=True)
    y = torch.from_numpy(audio_np.T.astype("float32")).mean(dim=0, keepdim=True)
    if sr != 44100:
        y = torchaudio.functional.resample(y, sr, 44100)
    # Take only first 3 seconds
    y = y[:, :44100*3]
    mel_spec = amp_to_db(mel_transform(y)).squeeze(0).numpy()
    return mel_spec

# Get random Groove file
import pandas as pd
import os
df = pd.read_csv('data/groove/info.csv')
groove_path = os.path.join('data/groove', df.iloc[0]['audio_filename'])
spec_groove = get_spec(groove_path)

spec_demucs = get_spec('output/stems/htdemucs/mp3uyU0HAQI/drums.wav')

fig, axs = plt.subplots(2, 1, figsize=(10, 8))
im1 = axs[0].imshow(spec_groove, aspect='auto', origin='lower')
axs[0].set_title('Groove Spectrogram (Training)')
fig.colorbar(im1, ax=axs[0])

im2 = axs[1].imshow(spec_demucs, aspect='auto', origin='lower')
axs[1].set_title('HTDemucs Spectrogram (Inference)')
fig.colorbar(im2, ax=axs[1])

plt.tight_layout()
plt.savefig('spectrogram_comparison.png')
print(f"Groove stats - min: {spec_groove.min():.2f}, max: {spec_groove.max():.2f}, mean: {spec_groove.mean():.2f}")
print(f"Demucs stats - min: {spec_demucs.min():.2f}, max: {spec_demucs.max():.2f}, mean: {spec_demucs.mean():.2f}")
