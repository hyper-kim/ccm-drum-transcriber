with open('src/transcriber.py', 'r', encoding='utf-8') as f:
    text = f.read()

# Restore peak normalization but apply it correctly with top_db logic like librosa
target = '''    if y.shape[0] > 1:
        y = y.mean(dim=0, keepdim=True)
        
    mel_transform = torchaudio.transforms.MelSpectrogram('''

replacement = '''    if y.shape[0] > 1:
        y = y.mean(dim=0, keepdim=True)
        
    # Scale audio so it matches the Groove dataset's peak volume
    y = y / torch.max(torch.abs(y))
    
    mel_transform = torchaudio.transforms.MelSpectrogram('''

text = text.replace(target, replacement)

target2 = '''onset_threshold: float = 0.02,'''
replacement2 = '''onset_threshold: float = 0.2,'''
text = text.replace(target2, replacement2)

with open('src/transcriber.py', 'w', encoding='utf-8') as f:
    f.write(text)
print("Restored peak normalization and raised threshold to 0.2")
