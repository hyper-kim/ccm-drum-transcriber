import re

with open("src/train/dataset.py", "r", encoding="utf-8") as f:
    text = f.read()

# We need to rewrite __getitem__ in GrooveDrumDataset to slice the audio BEFORE resampling and reading
# Let's find GrooveDrumDataset class
class_start = text.find('class GrooveDrumDataset')

old_getitem = '''    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        row_idx = idx % len(self.metadata)
        row = self.metadata.iloc[row_idx]
        
        audio_path = self.data_dir / row['audio_filename']
        midi_path = self.data_dir / row['midi_filename']
        
        audio, sr = sf.read(str(audio_path), always_2d=True)
        waveform = torch.from_numpy(audio.T.astype("float32")).mean(dim=0, keepdim=True)
        if sr != self.sr:
            waveform = torchaudio.functional.resample(waveform, sr, self.sr)
            
        duration_sec = waveform.shape[1] / self.sr
        roll = parse_midi_onsets(str(midi_path), duration_sec, fps=self.fps)
        
        max_start_sec = max(0, duration_sec - self.segment_sec)
        start_sec = np.random.uniform(0, max_start_sec)
        
        start_sample = int(start_sec * self.sr)
        end_sample = start_sample + int(self.segment_sec * self.sr)
        start_frame = int(start_sec * self.fps)
        end_frame = start_frame + self.segment_frames
        
        audio_segment = waveform[:, start_sample:end_sample]'''

new_getitem = '''    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        row_idx = idx % len(self.metadata)
        row = self.metadata.iloc[row_idx]
        
        audio_path = self.data_dir / row['audio_filename']
        midi_path = self.data_dir / row['midi_filename']
        
        info = sf.info(str(audio_path))
        duration_sec = info.frames / info.samplerate
        
        max_start_sec = max(0, duration_sec - self.segment_sec)
        start_sec = np.random.uniform(0, max_start_sec)
        
        # Read only the 3-second segment!
        start_frame_audio = int(start_sec * info.samplerate)
        num_frames = int(self.segment_sec * info.samplerate)
        
        audio, sr = sf.read(str(audio_path), start=start_frame_audio, frames=num_frames, always_2d=True)
        waveform = torch.from_numpy(audio.T.astype("float32")).mean(dim=0, keepdim=True)
        if sr != self.sr:
            waveform = torchaudio.functional.resample(waveform, sr, self.sr)
            
        # Parse MIDI
        roll = parse_midi_onsets(str(midi_path), duration_sec, fps=self.fps)
        
        start_frame = int(start_sec * self.fps)
        end_frame = start_frame + self.segment_frames
        
        audio_segment = waveform'''

text = text.replace(old_getitem, new_getitem)

with open("src/train/dataset.py", "w", encoding="utf-8") as f:
    f.write(text)

print("Dataset __getitem__ optimized!")
