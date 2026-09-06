import torch
import torchaudio
import soundfile as sf
import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal

from src.transcriber import DrumTranscriber, TranscriberConfig
import logging

logging.basicConfig(level=logging.INFO)

config = TranscriberConfig(model_path='models/model_stage3_best.pth')
transcriber = DrumTranscriber(config)

def test_inference():
    audio, sr = sf.read('output/stems/htdemucs/mp3uyU0HAQI/drums.wav', always_2d=True)
    y = torch.from_numpy(audio.T.astype("float32")).mean(dim=0, keepdim=True)
    if sr != transcriber.sr:
        y = torchaudio.functional.resample(y, sr, transcriber.sr)

    device = transcriber.device
    model = transcriber.model
    model.eval()

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=transcriber.sr, n_fft=2048, hop_length=transcriber.hop_length, 
        n_mels=229, f_min=30.0, f_max=transcriber.sr / 2.0
    ).to(device)
    amp_to_db = torchaudio.transforms.AmplitudeToDB().to(device)
    
    y_dev = y.to(device)
    mel_spec_raw = mel_transform(y_dev) # [1, 229, T]
    
    mel_spec = torch.zeros_like(mel_spec_raw)
    segment_frames = 300
    for i in range(0, mel_spec_raw.shape[-1], segment_frames):
        end_idx = min(i + segment_frames, mel_spec_raw.shape[-1])
        chunk = mel_spec_raw[..., i:end_idx]
        mel_spec[..., i:end_idx] = amp_to_db(chunk)
        
    mel_spec = mel_spec.unsqueeze(0) # [1, 1, 229, T]
    
    probs = np.zeros((mel_spec.shape[3], 11), dtype=np.float32)
    step_frames = 150
    total_frames = mel_spec.shape[3]
    
    with torch.no_grad():
        for start_f in range(0, total_frames, step_frames):
            end_f = min(start_f + segment_frames, total_frames)
            seg_len = end_f - start_f
            
            x = mel_spec[:, :, :, start_f:end_f]
            if x.shape[3] < segment_frames:
                pad = segment_frames - x.shape[3]
                x = torch.nn.functional.pad(x, (0, pad))
                
            logits = model(x)
            seg_probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
            
            probs[start_f:end_f] = np.maximum(probs[start_f:end_f], seg_probs[:seg_len])
            
    # Print hits
    for cls_idx in range(11):
        cls_probs = probs[:, cls_idx]
        peaks, _ = scipy.signal.find_peaks(cls_probs, height=0.5, distance=5)
        print(f"Class {cls_idx} max prob: {cls_probs.max():.4f}, peaks found (th=0.5): {len(peaks)}")

test_inference()
