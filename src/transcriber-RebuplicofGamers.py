"""
src/transcriber.py  (v2)

2단계 파이프라인:
  1. Onset detection  — librosa spectral flux (학습 불필요)
  2. Hit classification — DrumHitClassifier (100ms mel → class)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import List

import librosa
import mido
import numpy as np
import soundfile as sf
import torch
import torchaudio

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MIDI Pitch Map
# ---------------------------------------------------------------------------

class DrumPitch(IntEnum):
    BD        = 36
    SD        = 38
    HH_CLOSED = 42
    HH_OPEN   = 46
    TOM_H     = 48
    TOM_M     = 45
    TOM_F     = 41
    CRASH     = 49
    RIDE      = 51


# class index → representative MIDI pitch
_CLASS_TO_PITCH = {
    0: DrumPitch.BD,
    1: DrumPitch.SD,
    2: DrumPitch.HH_CLOSED,
    3: DrumPitch.HH_OPEN,
    4: DrumPitch.TOM_H,
    5: DrumPitch.TOM_M,
    6: DrumPitch.TOM_F,
    7: DrumPitch.CRASH,
    8: DrumPitch.RIDE,
}


@dataclass
class DrumEvent:
    time_sec:   float
    pitch:      DrumPitch
    velocity:   int
    confidence: float = 1.0


# ---------------------------------------------------------------------------
# MIDI export
# ---------------------------------------------------------------------------

def _export_midi(events: List[DrumEvent], out_path: Path,
                 bpm: float = 120.0) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mid   = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm), time=0))

    tpb = mid.ticks_per_beat

    def s2t(sec):
        return int(round(sec * bpm / 60.0 * tpb))

    cur = 0
    for ev in sorted(events, key=lambda e: e.time_sec):
        at  = s2t(ev.time_sec)
        dt  = max(0, at - cur)
        track.append(mido.Message("note_on",  channel=9,
                                  note=ev.pitch.value, velocity=ev.velocity, time=dt))
        track.append(mido.Message("note_off", channel=9,
                                  note=ev.pitch.value, velocity=0, time=10))
        cur = at + 10

    mid.save(str(out_path))
    logger.info(f"[Transcriber] MIDI saved → {out_path}")


# ---------------------------------------------------------------------------
# Onset detection
# ---------------------------------------------------------------------------

def _detect_onsets(waveform_np: np.ndarray, sr: int,
                   min_gap_sec: float = 0.05) -> np.ndarray:
    """
    Spectral flux onset detection via librosa.
    Returns onset times in seconds.
    """
    onset_frames = librosa.onset.onset_detect(
        y=waveform_np,
        sr=sr,
        hop_length=441,
        backtrack=True,
        units="frames",
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=441)

    # Merge onsets closer than min_gap_sec
    if len(onset_times) == 0:
        return onset_times
    merged = [onset_times[0]]
    for t in onset_times[1:]:
        if t - merged[-1] >= min_gap_sec:
            merged.append(t)
    return np.array(merged)


# ---------------------------------------------------------------------------
# Main transcription function
# ---------------------------------------------------------------------------

def transcribe_drums(
    drums_path: Path,
    *,
    min_gap_sec:      float = 0.05,
    confidence_thr:   float = 0.4,
    bpm_hint:         float = 120.0,
) -> List[DrumEvent]:
    """
    drums.wav (Demucs 출력) → DrumEvent list + output/drums.mid

    Pipeline:
      1. Onset detection (librosa spectral flux)
      2. Per-onset 100ms clip → mel → DrumHitClassifier
      3. Filter low-confidence predictions
    """
    from .train.dataset import (
        DrumClassMapping, SR, WINDOW_SAMPLES,
        build_mel_transform, wav_to_mel,
    )
    from .train.model import DrumHitClassifier

    logger.info(f"[Transcriber] Input: {drums_path}")

    model_path = Path("models/best_drum_classifier.pt")
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Run training first:\n"
            "  python -m src.train.train"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = DrumHitClassifier(num_classes=DrumClassMapping.NUM_CLASSES).to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.eval()

    # ---- Load audio ----
    audio_np, orig_sr = sf.read(str(drums_path), always_2d=True, dtype="float32")
    mono_np = audio_np.mean(axis=1)   # [T] float32 numpy

    if orig_sr != SR:
        waveform_t = torchaudio.functional.resample(
            torch.from_numpy(mono_np).unsqueeze(0), orig_sr, SR
        )
        mono_np = waveform_t.squeeze(0).numpy()
        sr_eff  = SR
    else:
        sr_eff = SR

    logger.info(f"[Transcriber] Audio loaded: {len(mono_np)/sr_eff:.1f}s @ {sr_eff}Hz")

    # ---- Stage 1: Onset detection ----
    onset_times = _detect_onsets(mono_np, sr_eff, min_gap_sec)
    logger.info(f"[Transcriber] {len(onset_times)} onsets detected")

    if len(onset_times) == 0:
        logger.warning("[Transcriber] No onsets found — check audio file")
        return []

    # ---- Stage 2: Hit classification ----
    mel_transform = build_mel_transform(device)
    events: List[DrumEvent] = []

    # Batch classify for efficiency
    batch_mels = []
    for t in onset_times:
        start = max(0, int(t * sr_eff) - WINDOW_SAMPLES // 4)  # 25ms before onset
        end   = start + WINDOW_SAMPLES
        clip  = mono_np[start:end] if end <= len(mono_np) else np.pad(
            mono_np[start:], (0, end - len(mono_np))
        )
        wf  = torch.from_numpy(clip).unsqueeze(0)   # [1, T]
        mel = wav_to_mel(wf.to(device), mel_transform)  # [N_MELS, T_frames]
        batch_mels.append(mel.unsqueeze(0))              # [1, N_MELS, T_frames]

    with torch.no_grad():
        batch = torch.stack(batch_mels).to(device)      # [N, 1, N_MELS, T]
        logits = model(batch)                            # [N, NUM_CLASSES]
        probs  = torch.softmax(logits, dim=1).cpu().numpy()

    for i, (t, prob) in enumerate(zip(onset_times, probs)):
        cls_idx    = int(np.argmax(prob))
        confidence = float(prob[cls_idx])

        if confidence < confidence_thr:
            logger.debug(f"  t={t:.3f}s  skipped (conf={confidence:.3f})")
            continue

        pitch = _CLASS_TO_PITCH.get(cls_idx, DrumPitch.SD)
        vel   = int(40 + confidence * 87)

        logger.debug(
            f"  t={t:.3f}s  {DrumClassMapping.CLASS_NAMES[cls_idx]:10s} "
            f"conf={confidence:.3f}  vel={vel}"
        )
        events.append(DrumEvent(
            time_sec=float(t), pitch=pitch,
            velocity=max(1, min(127, vel)), confidence=confidence,
        ))

    events.sort(key=lambda e: e.time_sec)
    logger.info(f"[Transcriber] {len(events)} events after filtering")

    midi_out = Path("output/drums.mid")
    _export_midi(events, midi_out, bpm_hint)

    return events
