import torch
import torchaudio
import numpy as np
import soundfile as sf
import os
import glob
from src.train.dataset import GrooveDrumDataset

dataset = GrooveDrumDataset("data", split="validation")
mel_spec, label = dataset[0]

from src.train.model import DrumCRNN
from src.train.dataset import DrumClassMapping
device = torch.device('cuda')
model = DrumCRNN(num_classes=DrumClassMapping.NUM_CLASSES).to(device)
model.load_state_dict(torch.load('models/best_drum_crnn_11class.pt', map_location=device, weights_only=True))
model.eval()

with torch.no_grad():
    x = mel_spec.unsqueeze(0).to(device) # [1, 1, 229, 301]
    logits = model(x)
    probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

print("Max prob per class on Groove Sample:", probs.max(axis=0))
