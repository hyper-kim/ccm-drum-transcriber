import torch
import torchaudio
import numpy as np
import soundfile as sf
from src.train.model import DrumCRNN
from src.train.dataset import DrumClassMapping

device = torch.device('cuda')
model = DrumCRNN(num_classes=DrumClassMapping.NUM_CLASSES).to(device)
model.load_state_dict(torch.load('models/best_drum_crnn_11class.pt', map_location=device, weights_only=True))
model.eval()

audio_np, sr = sf.read('output/stems/htdemucs/wmrlWj3x4dM_trimmed/drums.wav', always_2d=True)
y = torch.from_numpy(audio_np.T.astype("float32")).mean(dim=0, keepdim=True)
y = torchaudio.functional.resample(y, sr, 44100)

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=44100, n_fft=2048, hop_length=441, n_mels=229, f_min=30.0, f_max=44100/2.0
).to(device)
amp_to_db = torchaudio.transforms.AmplitudeToDB().to(device)

y_dev = y.to(device)
mel_spec = amp_to_db(mel_transform(y_dev)).unsqueeze(0)

print("Mel spec shape:", mel_spec.shape, "min:", mel_spec.min().item(), "max:", mel_spec.max().item())

total_frames = mel_spec.shape[3]
probs = np.zeros((total_frames, DrumClassMapping.NUM_CLASSES), dtype=np.float32)

with torch.no_grad():
    for start_f in range(0, total_frames, 300):
        end_f = min(start_f + 300, total_frames)
        x = mel_spec[:, :, :, start_f:end_f]
        if x.shape[3] < 300:
            x = torch.nn.functional.pad(x, (0, 300 - x.shape[3]))
        
        logits = model(x)
        probs[start_f:end_f] = torch.sigmoid(logits).squeeze(0).cpu().numpy()[:end_f - start_f]

print("Max prob per class over ENTIRE song:", probs.max(axis=0))
