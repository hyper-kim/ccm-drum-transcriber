import torch
import torchaudio
import soundfile as sf

def get_spec(path):
    audio_np, sr = sf.read(path, always_2d=True)
    y = torch.from_numpy(audio_np.T.astype("float32")).mean(dim=0, keepdim=True)
    if sr != 44100:
        y = torchaudio.functional.resample(y, sr, 44100)
    
    # torchaudio AmplitudeToDB logic directly implemented
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=44100, n_fft=2048, hop_length=441, n_mels=229, f_min=30.0, f_max=44100 / 2.0
    )
    amp_to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)
    
    mel_spec = amp_to_db(mel_transform(y)).squeeze(0)
    return mel_spec

# Get specs
import pandas as pd
import os
df = pd.read_csv('data/groove/info.csv')
groove_path = os.path.join('data/groove', df.iloc[0]['audio_filename'])

spec_groove = get_spec(groove_path)
spec_demucs = get_spec('output/stems/htdemucs/mp3uyU0HAQI/drums.wav')

print(f"Groove Raw Spec - min: {spec_groove.min():.2f}, max: {spec_groove.max():.2f}")
print(f"Demucs Raw Spec - min: {spec_demucs.min():.2f}, max: {spec_demucs.max():.2f}")

# The issue is AmplitudeToDB uses max of the input to set the 0 dB reference by default!
# So Demucs, which is quiet, gets referenced to its OWN quiet max. 
