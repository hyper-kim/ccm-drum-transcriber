import torch
import torchaudio
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from src.train.model import DrumCRNN
from src.train.dataset import DrumClassMapping

device = torch.device('cuda')
model = DrumCRNN(num_classes=DrumClassMapping.NUM_CLASSES).to(device)
model.load_state_dict(torch.load('models/best_drum_crnn_11class.pt', map_location=device, weights_only=True))
model.eval()

audio_np, sr = sf.read('output/stems/htdemucs/mp3uyU0HAQI/drums.wav', always_2d=True)
y = torch.from_numpy(audio_np.T.astype("float32")).mean(dim=0, keepdim=True)
y = torchaudio.functional.resample(y, sr, 44100)

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=44100, n_fft=2048, hop_length=441, n_mels=229, f_min=30.0, f_max=44100/2.0
).to(device)
amp_to_db = torchaudio.transforms.AmplitudeToDB().to(device)

y_dev = y[:, :44100*10].to(device) # first 10 seconds
mel_spec = amp_to_db(mel_transform(y_dev)).unsqueeze(0)

print("Mel spec shape:", mel_spec.shape, "min:", mel_spec.min().item(), "max:", mel_spec.max().item())

with torch.no_grad():
    x = mel_spec[:, :, :, :300]
    logits = model(x)
    probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

print("Probs max per class:", probs.max(axis=0))
print("Any prob > 0.5:", (probs > 0.5).any())
