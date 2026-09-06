with open('src/transcriber.py', 'r', encoding='utf-8') as f:
    text = f.read()

target = '''    amp_to_db = torchaudio.transforms.AmplitudeToDB().to(device)
    
    y_dev = y.to(device)
    mel_spec = amp_to_db(mel_transform(y_dev)).unsqueeze(0) # [1, 1, 229, T]'''

replacement = '''    amp_to_db = torchaudio.transforms.AmplitudeToDB().to(device)
    
    y_dev = y.to(device)
    mel_spec_raw = mel_transform(y_dev)
    
    # CRITICAL DOMAIN SHIFT FIX
    # The Demucs output is inherently quieter and has different variance than the Groove dataset.
    # We must match the Demucs statistics to the Groove dataset statistics BEFORE AmplitudeToDB.
    # From Groove dataset (train): mean ~ 23.0, std ~ 680.0
    # From Demucs (infer): mean ~ 2.5, std ~ 80.0
    mean_g, std_g = 23.0, 680.0
    mean_d, std_d = mel_spec_raw.mean(), mel_spec_raw.std()
    
    norm_demucs = (mel_spec_raw - mean_d) / (std_d + 1e-6)
    mel_spec_raw = torch.clamp(norm_demucs * std_g + mean_g, min=1e-10)
    
    mel_spec = amp_to_db(mel_spec_raw).unsqueeze(0) # [1, 1, 229, T]'''

text = text.replace(target, replacement)

with open('src/transcriber.py', 'w', encoding='utf-8') as f:
    f.write(text)
print("Applied z-score statistical matching")
