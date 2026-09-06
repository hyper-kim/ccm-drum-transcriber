import numpy as np
import soundfile as sf
import os
from pathlib import Path

def generate_applause(duration=10.0, sr=44100):
    # Generate pink noise (which sounds similar to clapping/rain)
    num_samples = int(duration * sr)
    white = np.random.randn(num_samples).astype(np.float32)
    # Simple lowpass to make white noise into pink-ish
    b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
    a = [1, -2.494956002,   2.017265875, -0.522189400]
    import scipy.signal
    pink = scipy.signal.lfilter(b, a, white)
    
    # Simulate individual claps (transients)
    num_claps = int(duration * 20) # 20 claps per second
    clap_indices = np.random.randint(0, num_samples - 1000, num_claps)
    for idx in clap_indices:
        env = np.exp(-np.linspace(0, 10, 1000))
        pink[idx:idx+1000] += env * np.random.randn(1000) * 0.5
        
    # Normalize
    pink = pink / np.max(np.abs(pink))
    
    Path("data").mkdir(exist_ok=True)
    sf.write("data/applause.wav", pink, sr)
    print("Generated data/applause.wav")

generate_applause()
