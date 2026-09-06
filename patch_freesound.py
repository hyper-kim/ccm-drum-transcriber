import re

with open("src/train/dataset.py", "r", encoding="utf-8") as f:
    text = f.read()

freesound_code = '''
class FreesoundDrumDataset(Dataset):
    """
    Freesound One-Shot Percussive Sounds Dataset (via mirdata).
    Uses keyword matching on tags to map to the 11-Class DrumPitch.
    """
    def __init__(self, data_dir: Path, sr: int = 44100, window_sec: float = 0.5):
        super().__init__()
        self.sr = sr
        self.window_frames = int(window_sec * sr)
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=2048, hop_length=441, n_mels=229, f_min=30.0, f_max=sr / 2.0
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        
        # Load Clapping Noise (Data Augmentation)
        noise_path = Path("data/applause.wav")
        if noise_path.exists():
            import soundfile as sf
            noise_audio, noise_sr = sf.read(str(noise_path), always_2d=True)
            self.clapping_noise = torch.from_numpy(noise_audio.T.astype("float32")).mean(dim=0, keepdim=True)
            if noise_sr != self.sr:
                self.clapping_noise = torchaudio.functional.resample(self.clapping_noise, noise_sr, self.sr)
        else:
            self.clapping_noise = None
            
        import mirdata
        try:
            self.dataset = mirdata.initialize("freesound_one_shot_percussive_sounds")
            self.tracks = self.dataset.load_tracks()
        except:
            self.tracks = {}
            
        self.samples = []
        self._parse_tracks()
        
    def _parse_tracks(self):
        # Keyword mapping to DrumPitch
        # DrumPitch classes: 0: BD, 1: RIM, 2: SD, 3: TOM_F, 4: HH_CLOSED, 5: HH_FOOT, 6: TOM_M, 7: HH_OPEN, 8: TOM_H, 9: CRASH, 10: RIDE, 11: RIDE_BELL, 12: SPLASH, 13: CRASH_2
        keyword_map = {
            "kick": 0, "bass-drum": 0,
            "snare": 2, "rimshot": 1,
            "hi-hat": 4, "hihat": 4, "closed-hi-hat": 4,
            "open-hi-hat": 7,
            "tom": 6, "low-tom": 3, "high-tom": 8, "floor-tom": 3,
            "crash": 9, "cymbal": 9,
            "ride": 10,
            "splash": 12
        }
        
        for track_id, track in self.tracks.items():
            if not track.tags:
                continue
            
            # Find matching class
            matched_cls = -1
            for tag in track.tags:
                tag_lower = tag.lower()
                for kw, cls_idx in keyword_map.items():
                    if kw in tag_lower:
                        matched_cls = cls_idx
                        break
                if matched_cls != -1:
                    break
                    
            if matched_cls != -1 and matched_cls in DrumClassMapping.MIDI_MAP.values():
                self.samples.append((track.audio_path, matched_cls))
                
        print(f"[FreesoundDataset] Parsed {len(self.samples)} valid drum hits from tags.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        audio_path, cls_idx = self.samples[idx]
        import soundfile as sf
        audio_np, orig_sr = sf.read(audio_path, always_2d=True)
        waveform = torch.from_numpy(audio_np.T.astype("float32")).mean(dim=0, keepdim=True)
        
        if orig_sr != self.sr:
            waveform = torchaudio.functional.resample(waveform, orig_sr, self.sr)
            
        if waveform.shape[1] > self.window_frames:
            waveform = waveform[:, :self.window_frames]
        elif waveform.shape[1] < self.window_frames:
            pad = self.window_frames - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad))
            
        # Data Augmentation: Clapping Noise
        if self.clapping_noise is not None and np.random.rand() < 0.3:
            noise_start = np.random.randint(0, max(1, self.clapping_noise.shape[1] - self.window_frames))
            noise_segment = self.clapping_noise[:, noise_start:noise_start+self.window_frames]
            if noise_segment.shape[1] < self.window_frames:
                noise_segment = torch.nn.functional.pad(noise_segment, (0, self.window_frames - noise_segment.shape[1]))
            gain = np.random.uniform(0.1, 0.5)
            waveform = waveform + noise_segment * gain
            
        mel_spec = self.amplitude_to_db(self.mel_transform(waveform)).squeeze(0)
        
        label = torch.zeros(DrumClassMapping.NUM_CLASSES, dtype=torch.float32)
        label[cls_idx] = 1.0
        
        return mel_spec, label
'''

if "FreesoundDrumDataset" not in text:
    text += "\n" + freesound_code

with open("src/train/dataset.py", "w", encoding="utf-8") as f:
    f.write(text)

print("FreesoundDrumDataset added to dataset.py")
