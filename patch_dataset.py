import re

with open("src/train/dataset.py", "r", encoding="utf-8") as f:
    text = f.read()

# We need to add the clapping noise augmentation logic.
# It should be loaded once in the Dataset __init__ and applied in __getitem__.

old_init = '''        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        
        df = pd.read_csv(self.data_dir / "info.csv")'''
        
new_init = '''        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        
        # Load clapping noise for data augmentation
        noise_path = Path("data/applause.wav")
        if noise_path.exists():
            noise_audio, noise_sr = sf.read(str(noise_path), always_2d=True)
            self.clapping_noise = torch.from_numpy(noise_audio.T.astype("float32")).mean(dim=0, keepdim=True)
            if noise_sr != self.sr:
                self.clapping_noise = torchaudio.functional.resample(self.clapping_noise, noise_sr, self.sr)
        else:
            self.clapping_noise = None
            
        df = pd.read_csv(self.data_dir / "info.csv")'''

old_getitem = '''        if waveform.shape[1] < self.window_frames:
            pad = self.window_frames - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad))
            
        mel_spec = self.amplitude_to_db(self.mel_transform(waveform)).squeeze(0)'''

new_getitem = '''        if waveform.shape[1] < self.window_frames:
            pad = self.window_frames - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad))
            
        # Data Augmentation: Add clapping noise 30% of the time
        if hasattr(self, 'clapping_noise') and self.clapping_noise is not None and np.random.rand() < 0.3:
            noise_start = np.random.randint(0, max(1, self.clapping_noise.shape[1] - self.window_frames))
            noise_segment = self.clapping_noise[:, noise_start:noise_start+self.window_frames]
            if noise_segment.shape[1] < self.window_frames:
                noise_segment = torch.nn.functional.pad(noise_segment, (0, self.window_frames - noise_segment.shape[1]))
            # Mix with random gain
            gain = np.random.uniform(0.1, 0.5)
            waveform = waveform + noise_segment * gain
            
        mel_spec = self.amplitude_to_db(self.mel_transform(waveform)).squeeze(0)'''

text = text.replace(old_init, new_init)
text = text.replace(old_getitem, new_getitem)

# Stage 3 getitem has a different window variable name (segment_frames)
old_getitem3 = '''        if audio_segment.shape[1] < int(self.segment_sec * self.sr):
            pad_len = int(self.segment_sec * self.sr) - audio_segment.shape[1]
            audio_segment = torch.nn.functional.pad(audio_segment, (0, pad_len))
            
        mel_spec = self.amplitude_to_db(self.mel_transform(audio_segment)).squeeze(0)'''

new_getitem3 = '''        if audio_segment.shape[1] < int(self.segment_sec * self.sr):
            pad_len = int(self.segment_sec * self.sr) - audio_segment.shape[1]
            audio_segment = torch.nn.functional.pad(audio_segment, (0, pad_len))
            
        # Data Augmentation for Stage 3
        if hasattr(self, 'clapping_noise') and self.clapping_noise is not None and np.random.rand() < 0.3:
            noise_start = np.random.randint(0, max(1, self.clapping_noise.shape[1] - self.segment_frames * self.hop_length))
            noise_segment = self.clapping_noise[:, noise_start:noise_start+audio_segment.shape[1]]
            if noise_segment.shape[1] < audio_segment.shape[1]:
                noise_segment = torch.nn.functional.pad(noise_segment, (0, audio_segment.shape[1] - noise_segment.shape[1]))
            gain = np.random.uniform(0.1, 0.5)
            audio_segment = audio_segment + noise_segment * gain
            
        mel_spec = self.amplitude_to_db(self.mel_transform(audio_segment)).squeeze(0)'''

text = text.replace(old_getitem3, new_getitem3)

with open("src/train/dataset.py", "w", encoding="utf-8") as f:
    f.write(text)

print("dataset.py patched with clapping noise augmentation.")
