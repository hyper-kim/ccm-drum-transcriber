import soundfile as sf
import numpy as np

# Groove dataset example
audio1, _ = sf.read(r'data\groove\drummer1\eval\session1\1_funk-groove1_138_beat_4-4.wav')
peak1 = np.max(np.abs(audio1))

# HTDemucs example
audio2, _ = sf.read(r'output\stems\htdemucs\mp3uyU0HAQI\drums.wav')
peak2 = np.max(np.abs(audio2))

print(f"Groove Peak: {peak1:.4f}")
print(f"HTDemucs Peak: {peak2:.4f}")
