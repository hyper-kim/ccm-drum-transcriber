import torch
import torchaudio
import numpy as np
import soundfile as sf
import scipy.signal
from src.train.model import DrumCRNN
from src.train.dataset import DrumClassMapping

device = torch.device('cuda')
model = DrumCRNN(num_classes=DrumClassMapping.NUM_CLASSES).to(device)
model.load_state_dict(torch.load('models/best_drum_crnn_11class.pt', map_location=device, weights_only=True))
model.eval()

audio_np, sr = sf.read('output/stems/htdemucs/wmrlWj3x4dM_trimmed/drums.wav', always_2d=True)
y = torch.from_numpy(audio_np.T.astype("float32")).mean(dim=0, keepdim=True)

# PEAK NORMALIZATION!
max_val = torch.max(torch.abs(y))
if max_val > 0:
    y = y / max_val

y = torchaudio.functional.resample(y, sr, 44100)

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=44100, n_fft=2048, hop_length=441, n_mels=229, f_min=30.0, f_max=44100/2.0
).to(device)
amp_to_db = torchaudio.transforms.AmplitudeToDB().to(device)

y_dev = y.to(device)
mel_spec = amp_to_db(mel_transform(y_dev)).unsqueeze(0)

total_frames = mel_spec.shape[3]
probs = np.zeros((total_frames, DrumClassMapping.NUM_CLASSES), dtype=np.float32)

with torch.no_grad():
    for start_f in range(0, total_frames, 150):
        end_f = min(start_f + 300, total_frames)
        x = mel_spec[:, :, :, start_f:end_f]
        if x.shape[3] < 300:
            x = torch.nn.functional.pad(x, (0, 300 - x.shape[3]))
        
        logits = model(x)
        seg_probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
        probs[start_f:end_f] = np.maximum(probs[start_f:end_f], seg_probs[:end_f - start_f])

for cls_idx in range(DrumClassMapping.NUM_CLASSES):
    peaks, _ = scipy.signal.find_peaks(probs[:, cls_idx], height=0.2, distance=5)
    print(f"Class {cls_idx} max prob: {probs[:, cls_idx].max():.4f}, peaks > 0.2: {len(peaks)}")
