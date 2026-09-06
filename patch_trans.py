with open('src/transcriber.py', 'r', encoding='utf-8') as f:
    text = f.read()

target = '''    if y.shape[0] > 1:
        y = y.mean(dim=0, keepdim=True)
        
    mel_transform = torchaudio.transforms.MelSpectrogram('''

replacement = '''    if y.shape[0] > 1:
        y = y.mean(dim=0, keepdim=True)
        
    max_val = torch.max(torch.abs(y))
    if max_val > 0:
        y = y / max_val
        
    mel_transform = torchaudio.transforms.MelSpectrogram('''

text = text.replace(target, replacement)

with open('src/transcriber.py', 'w', encoding='utf-8') as f:
    f.write(text)
print("Added peak normalization!")
