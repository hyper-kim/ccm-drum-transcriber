with open('src/transcriber.py', 'r', encoding='utf-8') as f:
    text = f.read()

target = '''    y_dev = y.to(device)
    mel_spec = amp_to_db(mel_transform(y_dev)).unsqueeze(0) # [1, 1, 229, T]'''

replacement = '''    y_dev = y.to(device)
    mel_spec = mel_transform(y_dev)
    # MANUALLY do amplitude to db so we have a FIXED reference of 1.0 (0 dB FS) for both train and infer.
    # torchaudio's AmplitudeToDB uses max(spec) as the reference, which breaks when audio is quiet!
    mel_spec = 10.0 * torch.log10(torch.clamp(mel_spec, min=1e-10))
    # We still need to bound it to top_db to match training behavior, but with an absolute reference.
    # Actually, during training, dataset.py used AmplitudeToDB which used the MAX OF EACH CHUNK.
    # To EXACTLY replicate training, we must apply AmplitudeToDB per 300-frame chunk, not globally!
    mel_spec = amp_to_db(mel_spec).unsqueeze(0)
'''

text = text.replace(target, replacement)

with open('src/transcriber.py', 'w', encoding='utf-8') as f:
    f.write(text)
print("Updated manual DB conversion")
