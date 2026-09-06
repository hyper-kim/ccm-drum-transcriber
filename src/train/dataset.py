import os
import logging
from pathlib import Path
from typing import List, Tuple, Dict
import pandas as pd
import numpy as np
import torch
import torchaudio
import mido
import soundfile as sf
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

DATASET_URL = "https://storage.googleapis.com/magentadata/datasets/groove/groove-v1.0.0-midionly.zip"

class DrumClassMapping:
    """11-Class Mapping for High-Precision Transcription"""
    # 0: Kick
    # 1: Snare
    # 2: ClosedHH
    # 3: OpenHH
    # 4: SmallTom
    # 5: MidTom
    # 6: LargeTom
    # 7: Crash1
    # 8: Crash2
    # 9: Splash
    # 10: Ride
    
    MIDI_MAP = {
        36: 0, # Kick
        38: 1, 40: 1, 37: 1, # Snare (including Rimshot/Cross-stick)
        42: 2, 44: 2, # Closed HH / HH Pedal
        46: 3, # Open HH
        48: 4, 50: 4, # Small Tom / High Tom
        45: 5, 47: 5, # Mid Tom
        41: 6, 43: 6, # Large Tom / Floor Tom
        49: 7, # Crash 1
        57: 8, # Crash 2
        55: 9, # Splash
        51: 10, 53: 10, 59: 10, # Ride
    }
    NUM_CLASSES = 11

def download_and_extract(data_dir: Path):
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "groove-v1.0.0.zip"
    extract_dir = data_dir / "groove"
    
    if extract_dir.exists():
        logger.info(f"Dataset exists at {extract_dir}")
        return extract_dir
    logger.error("Dataset not found. Please extract the groove dataset.")
    return extract_dir


def parse_midi_onsets(midi_path: str, duration_sec: float, fps: int = 100) -> np.ndarray:
    mid = mido.MidiFile(midi_path)
    num_frames = int(np.ceil(duration_sec * fps))
    roll = np.zeros((num_frames, DrumClassMapping.NUM_CLASSES), dtype=np.float32)
    
    # Gaussian kernel for temporal label smoothing
    kernel = np.array([0.1, 0.5, 1.0, 0.5, 0.1], dtype=np.float32)
    kernel_radius = len(kernel) // 2
    
    current_time = 0.0
    for track in mid.tracks:
        for msg in track:
            current_time += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                if getattr(msg, 'channel', 9) == 9: # Drum Channel
                    if msg.note in DrumClassMapping.MIDI_MAP:
                        cls_idx = DrumClassMapping.MIDI_MAP[msg.note]
                        frame_idx = int(round(current_time * fps))
                        
                        # Apply label smoothing
                        for i, val in enumerate(kernel):
                            target_frame = frame_idx - kernel_radius + i
                            if 0 <= target_frame < num_frames:
                                roll[target_frame, cls_idx] = max(roll[target_frame, cls_idx], val)
    return roll


class IsolatedGrooveDataset(Dataset):
    """Stage 1: Curriculum Learning - Isolated Drum Hits (Single-label)"""
    def __init__(self, data_dir: str, split: str = 'train', sr: int = 44100, window_ms: int = 200):
        self.data_dir = Path(data_dir) / "groove"
        self.sr = sr
        self.window_frames = int((window_ms / 1000.0) * sr)
        
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=2048, hop_length=441, n_mels=229, f_min=30.0, f_max=sr / 2.0
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        
        # Load clapping noise for data augmentation
        noise_path = Path("data/applause.wav")
        if noise_path.exists():
            noise_audio, noise_sr = sf.read(str(noise_path), always_2d=True)
            self.clapping_noise = torch.from_numpy(noise_audio.T.astype("float32")).mean(dim=0, keepdim=True)
            if noise_sr != self.sr:
                self.clapping_noise = torchaudio.functional.resample(self.clapping_noise, noise_sr, self.sr)
        else:
            self.clapping_noise = None
            
        df = pd.read_csv(self.data_dir / "info.csv")
        df = df[(df['split'] == split) & (df['audio_filename'].notna())]
        
        self.samples = []
        logger.info(f"Scanning for isolated hits in {split} split...")
        
        for _, row in df.iterrows():
            audio_path = self.data_dir / row['audio_filename']
            midi_path = self.data_dir / row['midi_filename']
            if not audio_path.exists() or not midi_path.exists():
                continue
                
            mid = mido.MidiFile(midi_path)
            notes = []
            curr_time = 0.0
            for track in mid.tracks:
                for msg in track:
                    curr_time += msg.time
                    if msg.type == 'note_on' and msg.velocity > 0 and getattr(msg, 'channel', 9) == 9:
                        if msg.note in DrumClassMapping.MIDI_MAP:
                            notes.append({'time': curr_time, 'class': DrumClassMapping.MIDI_MAP[msg.note]})
            
            notes.sort(key=lambda x: x['time'])
            
            for i in range(len(notes)):
                n = notes[i]
                is_isolated = True
                if i > 0 and n['time'] - notes[i-1]['time'] < 0.1:
                    is_isolated = False
                if i < len(notes)-1 and notes[i+1]['time'] - n['time'] < 0.1:
                    is_isolated = False
                    
                if is_isolated:
                    self.samples.append({
                        'audio_path': audio_path,
                        'time': n['time'],
                        'class': n['class']
                    })
        
        logger.info(f"Found {len(self.samples)} isolated hits for {split}.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        center_sample = int(sample['time'] * self.sr)
        start_sample = max(0, center_sample - self.window_frames // 2)
        
        audio, orig_sr = sf.read(str(sample['audio_path']), start=start_sample, frames=self.window_frames, always_2d=True)
        waveform = torch.from_numpy(audio.T.astype("float32")).mean(dim=0, keepdim=True)
        
        if orig_sr != self.sr:
            waveform = torchaudio.functional.resample(waveform, orig_sr, self.sr)
            
        if waveform.shape[1] < self.window_frames:
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
            
        mel_spec = self.amplitude_to_db(self.mel_transform(waveform)).squeeze(0)
        x = mel_spec.unsqueeze(0) # [1, 229, T]
        
        y = torch.zeros(DrumClassMapping.NUM_CLASSES, dtype=torch.float32)
        y[sample['class']] = 1.0
        
        return x, y


class SuperimposedGrooveDataset(Dataset):
    """Stage 2: Curriculum Learning - Superimposed/Overlapping Hits (Multi-label)"""
    def __init__(self, data_dir: str, split: str = 'train', sr: int = 44100, window_ms: int = 200):
        self.data_dir = Path(data_dir) / "groove"
        self.sr = sr
        self.window_frames = int((window_ms / 1000.0) * sr)
        
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=2048, hop_length=441, n_mels=229, f_min=30.0, f_max=sr / 2.0
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        
        # Load clapping noise for data augmentation
        noise_path = Path("data/applause.wav")
        if noise_path.exists():
            noise_audio, noise_sr = sf.read(str(noise_path), always_2d=True)
            self.clapping_noise = torch.from_numpy(noise_audio.T.astype("float32")).mean(dim=0, keepdim=True)
            if noise_sr != self.sr:
                self.clapping_noise = torchaudio.functional.resample(self.clapping_noise, noise_sr, self.sr)
        else:
            self.clapping_noise = None
            
        df = pd.read_csv(self.data_dir / "info.csv")
        df = df[(df['split'] == split) & (df['audio_filename'].notna())]
        
        self.samples = []
        logger.info(f"Scanning for superimposed hits in {split} split...")
        
        for _, row in df.iterrows():
            audio_path = self.data_dir / row['audio_filename']
            midi_path = self.data_dir / row['midi_filename']
            if not audio_path.exists() or not midi_path.exists():
                continue
                
            mid = mido.MidiFile(midi_path)
            notes = []
            curr_time = 0.0
            for track in mid.tracks:
                for msg in track:
                    curr_time += msg.time
                    if msg.type == 'note_on' and msg.velocity > 0 and getattr(msg, 'channel', 9) == 9:
                        if msg.note in DrumClassMapping.MIDI_MAP:
                            notes.append({'time': curr_time, 'class': DrumClassMapping.MIDI_MAP[msg.note]})
            
            notes.sort(key=lambda x: x['time'])
            
            # Group notes that occur within 20ms of each other
            i = 0
            while i < len(notes):
                group = [notes[i]]
                j = i + 1
                while j < len(notes) and notes[j]['time'] - notes[i]['time'] <= 0.02:
                    group.append(notes[j])
                    j += 1
                
                if len(group) > 1: # Only want overlapping sounds
                    # Check isolation from non-group notes
                    is_isolated = True
                    if i > 0 and group[0]['time'] - notes[i-1]['time'] < 0.1:
                        is_isolated = False
                    if j < len(notes) and notes[j]['time'] - group[-1]['time'] < 0.1:
                        is_isolated = False
                        
                    if is_isolated:
                        classes = set([n['class'] for n in group])
                        self.samples.append({
                            'audio_path': audio_path,
                            'time': group[0]['time'],
                            'classes': list(classes)
                        })
                i = j
                
        logger.info(f"Found {len(self.samples)} superimposed hits for {split}.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        center_sample = int(sample['time'] * self.sr)
        start_sample = max(0, center_sample - self.window_frames // 2)
        
        audio, orig_sr = sf.read(str(sample['audio_path']), start=start_sample, frames=self.window_frames, always_2d=True)
        waveform = torch.from_numpy(audio.T.astype("float32")).mean(dim=0, keepdim=True)
        
        if orig_sr != self.sr:
            waveform = torchaudio.functional.resample(waveform, orig_sr, self.sr)
            
        if waveform.shape[1] < self.window_frames:
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
            
        mel_spec = self.amplitude_to_db(self.mel_transform(waveform)).squeeze(0)
        x = mel_spec.unsqueeze(0)
        
        y = torch.zeros(DrumClassMapping.NUM_CLASSES, dtype=torch.float32)
        for cls in sample['classes']:
            y[cls] = 1.0
            
        return x, y

class GrooveDrumDataset(Dataset):
    """Stage 3: Full Context Transcription (Continuous Frame-wise)"""
    def __init__(self, data_dir: str, split: str = 'train', sr: int = 44100, hop_length: int = 441, segment_sec: float = 3.0):
        self.data_dir = Path(data_dir) / "groove"
        self.split = split
        self.sr = sr
        self.hop_length = hop_length
        self.fps = sr // hop_length
        self.segment_sec = segment_sec
        self.segment_frames = int(segment_sec * self.fps)
        
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=2048, hop_length=hop_length, n_mels=229, f_min=30.0, f_max=sr / 2.0
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        
        # Load clapping noise for data augmentation
        noise_path = Path("data/applause.wav")
        if noise_path.exists():
            noise_audio, noise_sr = sf.read(str(noise_path), always_2d=True)
            self.clapping_noise = torch.from_numpy(noise_audio.T.astype("float32")).mean(dim=0, keepdim=True)
            if noise_sr != self.sr:
                self.clapping_noise = torchaudio.functional.resample(self.clapping_noise, noise_sr, self.sr)
        else:
            self.clapping_noise = None
            
        df = pd.read_csv(self.data_dir / "info.csv")
        self.metadata = df[(df['split'] == split) & (df['audio_filename'].notna())].reset_index(drop=True)
        
    def __len__(self):
        return len(self.metadata) * 5
        
    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
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
        
        audio_segment = waveform
        if audio_segment.shape[1] < int(self.segment_sec * self.sr):
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
            
        mel_spec = self.amplitude_to_db(self.mel_transform(audio_segment)).squeeze(0)
        
        roll_segment = roll[start_frame:end_frame, :]
        if roll_segment.shape[0] < self.segment_frames:
            pad_len = self.segment_frames - roll_segment.shape[0]
            roll_segment = np.pad(roll_segment, ((0, pad_len), (0, 0)))
            
        x = mel_spec.unsqueeze(0)
        y = torch.from_numpy(roll_segment).float()
        
        return x, y


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
                if Path(track.audio_path).exists():
                    self.samples.append((track.audio_path, matched_cls))
                    
        print(f"[FreesoundDataset] Parsed {len(self.samples)} valid drum hits from mirdata tags.")

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
