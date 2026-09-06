with open('src/transcriber.py', 'r', encoding='utf-8') as f:
    text = f.read()

target = '''    mel_spec_raw = mel_transform(y_dev) # Shape: [1, 229, T]
    
    # torchaudio AmplitudeToDB uses the max of the input tensor.
    # If we apply it globally, the entire song gets scaled by the loudest single snare hit, 
    # making quiet ghost notes disappear into the -80dB floor.
    # In training (dataset.py), AmplitudeToDB was applied on 300-frame chunks individually!
    # So we MUST replicate that exact behavior here.
    mel_spec = torch.zeros_like(mel_spec_raw)
    segment_frames = 300
    for i in range(0, mel_spec_raw.shape[-1], segment_frames):
        end_idx = min(i + segment_frames, mel_spec_raw.shape[-1])
        chunk = mel_spec_raw[..., i:end_idx]
        mel_spec[..., i:end_idx] = amp_to_db(chunk)
        
    mel_spec = mel_spec.unsqueeze(0) # [1, 1, 229, T]'''

# The ultimate fix: Demucs output power is 9218 max, Groove is 67422 max.
# By forcing top_db=80 and calculating AmplitudeToDB WITHOUT relative scaling to the max chunk,
# or simply standardizing the spectrogram using (x - mean)/std BEFORE the model.
# Wait, the model uses BatchNorm2d right after the input. 
# So the absolute level doesn't matter! The model normalizes it across the batch!
# If BatchNorm2d is normalizing across the batch, during inference with batch size 1, 
# it uses running mean/var! We trained with batch size 384!
replacement = '''    mel_spec = amp_to_db(mel_transform(y_dev)).unsqueeze(0)'''

text = text.replace(target, replacement)

target2 = '''onset_threshold: float = 0.1,'''
replacement2 = '''onset_threshold: float = 0.5,'''
text = text.replace(target2, replacement2)

with open('src/transcriber.py', 'w', encoding='utf-8') as f:
    f.write(text)
print("Reverted to global amp_to_db")
