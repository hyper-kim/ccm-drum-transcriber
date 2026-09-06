import torch
import torchaudio
import numpy as np
import soundfile as sf

path = 'data/groove/drummer7/session3/98_hiphop_70_fill_4-4.wav'
try:
    audio, sr = sf.read(path, start=0, frames=44100*3, always_2d=True)
    waveform = torch.from_numpy(audio.T.astype("float32")).mean(dim=0, keepdim=True)

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=44100, n_fft=2048, hop_length=441, n_mels=229, f_min=30.0, f_max=44100/2.0
    )
    amp_to_db = torchaudio.transforms.AmplitudeToDB()

    mel_spec = amp_to_db(mel_transform(waveform))

    print(f"Groove Dataset Mel Spec -> min: {mel_spec.min().item()}, max: {mel_spec.max().item()}")
except Exception as e:
    print(f"Error: {e}")
