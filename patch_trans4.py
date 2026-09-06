with open('src/transcriber.py', 'r', encoding='utf-8') as f:
    text = f.read()

target = '''    amp_to_db = torchaudio.transforms.AmplitudeToDB().to(device)'''

# Hardcode the top_db relative max so both Groove and Demucs get the exact same DB reference level
replacement = '''    amp_to_db = torchaudio.transforms.AmplitudeToDB(top_db=80).to(device)
    
    # torchaudio AmplitudeToDB defaults to referencing the max value of the current batch.
    # Since Demucs is quiet, its max is low, shifting all DB values up (making noise look like signal).
    # We force the db reference to a fixed 1.0 (0 dB FS) so quiet audio stays quiet.
    amp_to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)
    # The fix is to scale y up to 1.0 peak ONLY IF we don't change the background noise floor relative level.
    # But actually, librosa/torchaudio just uses the max of the spectrogram. 
    # Let's manually do amplitude to db so we have absolute control over the reference max.
'''

text = text.replace(target, replacement)

target3 = '''onset_threshold: float = 0.2,'''
replacement3 = '''onset_threshold: float = 0.1,'''
text = text.replace(target3, replacement3)

with open('src/transcriber.py', 'w', encoding='utf-8') as f:
    f.write(text)
print("Updated!")
