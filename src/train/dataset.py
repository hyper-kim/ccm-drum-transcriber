"""
src/train/dataset.py  (v3)

MIDI 완전 폐기. 실제 드럼 hit 오디오 데이터셋으로 지도학습.

학습 데이터:
  1. FSD50K        — Freesound/AudioSet 레이블, 대규모
  2. GrooveOnset   — Groove audio + MIDI onset 타이밍 → 클리핑

디렉토리 구조:
  data/
    FSD50K.dev_audio/        (wav files)
    FSD50K.ground_truth/
      dev.csv
    groove/
      info.csv
      drummer*/...

입력 X: [1, N_MELS, T_frames] (100ms mel 스펙트로그램)
출력 Y: int (class index)
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import mido
import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import ConcatDataset, Dataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Class mapping
# ---------------------------------------------------------------------------

class DrumClassMapping:
    """11-class mapping based on timbral/frequency characteristics."""
    CLASS_NAMES = [
        "Kick",       # 0  — low freq, strong fundamental
        "Snare",      # 1  — mid + broadband noise
        "HH-Closed",  # 2  — high freq, short decay
        "HH-Open",    # 3  — high freq, long decay
        "Tom-H",      # 4  — high tom
        "Tom-M",      # 5  — mid tom
        "Tom-L",      # 6  — floor tom / large tom
        "Crash",      # 7  — broadband, long decay
        "Ride",       # 8  — mid-high, bow sound
        "Ride-Bell",  # 9  — ride bell, sharp ping (new)
        "Rimshot",    # 10 — sharp crack, mid-high (new)
    ]
    NUM_CLASSES = len(CLASS_NAMES)

    # FSD50K / AudioSet label strings → class index
    # 공백/언더스코어 형식 모두 포함
    FSD50K_MAP = {
        # 공백 형식
        "Bass drum":             0,
        "Kick drum":             0,
        "Snare drum":            1,
        "Hi-hat":                2,
        "Closed hi-hat":         2,
        "Open hi-hat":           3,
        "Tom-tom":               4,
        "High tom-tom":          4,
        "Mid tom-tom":           5,
        "Floor tom":             6,
        "Low tom-tom":           6,
        "Crash cymbal":          7,
        "Ride cymbal":           8,
        "Rimshot":               10,
        # 언더스코어 형식 (FSD50K dev.csv 실제 포맷)
        "Bass_drum":             0,
        "Kick_drum":             0,
        "Snare_drum":            1,
        "Closed_hi-hat":         2,
        "Open_hi-hat":           3,
        "Tom-tom":               4,
        "High_tom-tom":          4,
        "Mid_tom-tom":           5,
        "Floor_tom":             6,
        "Low_tom-tom":           6,
        "Crash_cymbal":          7,
        "Ride_cymbal":           8,
    }

    # Groove MIDI note → class index
    GROOVE_MIDI_MAP = {
        36: 0,               # Kick
        38: 1, 40: 1,        # Snare (acoustic, electric)
        37: 10,              # Rimshot → 별도 클래스
        42: 2, 44: 2,        # HH Closed / Pedal
        46: 3,               # HH Open
        48: 4, 50: 4,        # High Tom
        45: 5, 47: 5,        # Mid Tom
        41: 6, 43: 6,        # Floor Tom
        49: 7, 57: 7, 55: 7, # Crash / Splash / China
        51: 8, 59: 8,        # Ride (bow)
        53: 9,               # Ride Bell → 별도 클래스
    }


# ---------------------------------------------------------------------------
# Feature extraction helpers  (shared with transcriber.py)
# ---------------------------------------------------------------------------

SR             = 44100
N_FFT          = 2048
HOP_LENGTH     = 441       # 10 ms → 100 fps
N_MELS         = 128
F_MIN          = 20.0
F_MAX          = SR / 2.0
FPS            = SR // HOP_LENGTH   # 100
WINDOW_MS      = 200                # 분류 윈도우 200ms (HH-Open/Crash 구분을 위해 확장)
WINDOW_FRAMES  = int(WINDOW_MS / 1000 * FPS)   # 20 frames
WINDOW_SAMPLES = int(WINDOW_MS / 1000 * SR)    # 8820 samples


def build_mel_transform(device: torch.device = torch.device("cpu")):
    return torchaudio.transforms.MelSpectrogram(
        sample_rate=SR, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, f_min=F_MIN, f_max=F_MAX,
    ).to(device)


def normalize_mel(spec: torch.Tensor) -> torch.Tensor:
    """Per-sample z-score normalization. Applied identically at train AND inference."""
    mean = spec.mean()
    std  = spec.std() + 1e-8
    return (spec - mean) / std


def wav_to_mel(waveform: torch.Tensor,
               mel_transform: torchaudio.transforms.MelSpectrogram) -> torch.Tensor:
    """
    waveform: [1, T]  (mono, float32)
    returns:  [N_MELS, T_frames]  (normalized log-mel)
    """
    mel = mel_transform(waveform)
    mel_db = torchaudio.functional.amplitude_to_DB(
        mel, multiplier=10.0, amin=1e-10,
        db_multiplier=0.0, top_db=80.0,
    ).squeeze(0)
    return normalize_mel(mel_db)


def load_audio_clip(path: str | Path, start_sample: int,
                    n_samples: int, target_sr: int = SR) -> torch.Tensor:
    """Load mono audio clip, return [1, n_samples] tensor."""
    audio, orig_sr = sf.read(
        str(path), start=start_sample,
        frames=n_samples, always_2d=True,
        dtype="float32",
    )
    waveform = torch.from_numpy(audio.T).mean(0, keepdim=True)
    if orig_sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, orig_sr, target_sr)
    n = int(n_samples * target_sr / orig_sr)
    if waveform.shape[1] < n:
        waveform = torch.nn.functional.pad(waveform, (0, n - waveform.shape[1]))
    return waveform[:, :n]


def augment_waveform(waveform: torch.Tensor) -> torch.Tensor:
    """Light augmentation: random gain + polarity flip."""
    waveform = waveform * float(np.random.uniform(0.5, 1.5))
    if np.random.rand() < 0.5:
        waveform = -waveform
    return waveform.clamp(-1.0, 1.0)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class FSD50KDrumDataset(Dataset):
    """
    FSD50K — files placed directly under data/ (no subfolder).

      data/
        FSD50K.dev_audio/    (wav files)
        FSD50K.ground_truth/
          dev.csv
    """

    def __init__(self, data_dir: str | Path, split: str = "train",
                 augment: bool = True):
        self.data_dir = Path(data_dir)
        self.augment  = augment
        self.mel      = build_mel_transform()

        audio_dir  = self.data_dir / "FSD50K.dev_audio"
        labels_csv = self.data_dir / "FSD50K.ground_truth" / "dev.csv"

        if not labels_csv.exists():
            raise FileNotFoundError(
                f"FSD50K labels not found at {labels_csv}.\n"
                "Download from: https://zenodo.org/record/4060432"
            )

        self.samples: list[tuple[Path, int]] = []

        with open(labels_csv, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                split_tag = row.get("split", "train")
                if split == "train"      and split_tag != "train": continue
                if split == "validation" and split_tag != "val":   continue

                labels  = [l.strip() for l in row["labels"].split(",")]
                matched = [DrumClassMapping.FSD50K_MAP[l]
                           for l in labels if l in DrumClassMapping.FSD50K_MAP]

                if len(matched) == 1:
                    fpath = audio_dir / (row["fname"] + ".wav")
                    if fpath.exists():
                        self.samples.append((fpath, matched[0]))

        logger.info(f"[FSD50K] {split}: {len(self.samples)} drum clips")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> tuple[torch.Tensor, int]:
        path, cls = self.samples[idx]
        audio, orig_sr = sf.read(str(path), always_2d=True, dtype="float32")
        waveform = torch.from_numpy(audio.T).mean(0, keepdim=True)
        if orig_sr != SR:
            waveform = torchaudio.functional.resample(waveform, orig_sr, SR)
        if waveform.shape[1] < WINDOW_SAMPLES:
            waveform = torch.nn.functional.pad(
                waveform, (0, WINDOW_SAMPLES - waveform.shape[1]))
        else:
            waveform = waveform[:, :WINDOW_SAMPLES]
        if self.augment:
            waveform = augment_waveform(waveform)
        mel = wav_to_mel(waveform, self.mel)
        return mel.unsqueeze(0), cls


class GrooveOnsetDataset(Dataset):
    """
    Groove MIDI/Audio — MIDI onset을 위치 탐지기로만 사용, 오디오 스펙트럼으로 학습.

    data/groove/
      info.csv
      drummer*/session*/.../  (*.wav + *.mid)
    """

    def __init__(self, data_dir: str | Path, split: str = "train",
                 augment: bool = True):
        import mido as _mido
        import pandas as pd

        self.data_dir = Path(data_dir) / "groove"
        self.augment  = augment
        self.mel      = build_mel_transform()
        self.samples: list[tuple[Path, float, int]] = []

        try:
            df = pd.read_csv(self.data_dir / "info.csv")
        except FileNotFoundError:
            logger.warning(f"[GrooveOnset] info.csv not found at {self.data_dir}")
            return

        df = df[(df["split"] == split) & df["audio_filename"].notna()]

        for _, row in df.iterrows():
            audio_path = self.data_dir / row["audio_filename"]
            midi_path  = self.data_dir / row["midi_filename"]
            if not audio_path.exists() or not midi_path.exists():
                continue

            mid    = _mido.MidiFile(str(midi_path))
            tempo  = 500_000
            tpb    = mid.ticks_per_beat

            for track in mid.tracks:
                abs_ticks = 0
                for msg in track:
                    abs_ticks += msg.time
                    if msg.type == "set_tempo":
                        tempo = msg.tempo
                    if (msg.type == "note_on" and msg.velocity > 0
                            and getattr(msg, "channel", 9) == 9
                            and msg.note in DrumClassMapping.GROOVE_MIDI_MAP):
                        t_sec = _mido.tick2second(abs_ticks, tpb, tempo)
                        cls   = DrumClassMapping.GROOVE_MIDI_MAP[msg.note]
                        self.samples.append((audio_path, t_sec, cls))

        logger.info(f"[GrooveOnset] {split}: {len(self.samples)} onset clips")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> tuple[torch.Tensor, int]:
        wav_path, t_sec, cls = self.samples[idx]
        start    = max(0, int(t_sec * SR) - WINDOW_SAMPLES // 4)
        waveform = load_audio_clip(wav_path, start, WINDOW_SAMPLES)
        if self.augment:
            waveform = augment_waveform(waveform)
        mel = wav_to_mel(waveform, self.mel)
        return mel.unsqueeze(0), cls


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_dataset(data_dir: str | Path, split: str = "train") -> ConcatDataset:
    """사용 가능한 데이터셋 자동 합성. 없는 것은 경고 후 스킵."""
    data_dir = Path(data_dir)
    datasets = []
    augment  = (split == "train")

    for cls, name in [
        (FSD50KDrumDataset,  "FSD50K"),
        (GrooveOnsetDataset, "Groove"),
    ]:
        try:
            ds = cls(data_dir, split=split, augment=augment)
            if len(ds) > 0:
                datasets.append(ds)
                logger.info(f"  + {name}: {len(ds)} samples ({split})")
        except FileNotFoundError as e:
            logger.warning(f"  - {name} skipped: {e}")

    if not datasets:
        raise RuntimeError(
            "No datasets found! Download FSD50K or Groove audio first."
        )

    combined = ConcatDataset(datasets)
    logger.info(f"[Dataset] Total {split}: {len(combined)} samples")
    return combined
