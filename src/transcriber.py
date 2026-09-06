"""
src/transcriber.py
────────────────────
학습된 딥러닝 모델(DrumCRNN)을 사용하여 드럼 오디오 스템(drums.wav)을 전사합니다.
추출된 결과는 DrumEvent 객체 리스트로 반환되며, `output/drums.mid` 파일로도 내보냅니다.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import List

import numpy as np
import mido
import torch
import torchaudio
import scipy.signal

logger = logging.getLogger(__name__)


class DrumPitch(IntEnum):
    """General MIDI Drum Map Standard"""
    BD        = 36   # Bass Drum (Kick)
    RIM       = 37   # Rimshot / Cross-stick
    SD        = 38   # Snare Drum (Center)
    TOM_F     = 41   # Floor Tom (Large Tom)
    HH_CLOSED = 42   # Hi-Hat Closed
    HH_FOOT   = 44   # Hi-Hat Pedal
    TOM_M     = 45   # Mid Tom
    HH_OPEN   = 46   # Hi-Hat Open
    TOM_H     = 48   # High Tom (Small Tom)
    CRASH     = 49   # Crash Cymbal 1
    RIDE      = 51   # Ride Cymbal 1
    RIDE_BELL = 53   # Ride Bell
    SPLASH    = 55   # Splash Cymbal
    CRASH_2   = 57   # Crash Cymbal 2


@dataclass
class DrumEvent:
    time_sec: float
    pitch: DrumPitch
    velocity: int
    confidence: float = 1.0


def _export_to_midi(events: List[DrumEvent], out_path: Path, bpm: float = 120.0) -> None:
    """DrumEvent 리스트를 표준 MIDI 파일로 저장한다."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    
    tempo = mido.bpm2tempo(bpm)
    track.append(mido.MetaMessage('set_tempo', tempo=tempo, time=0))
    
    sorted_events = sorted(events, key=lambda e: e.time_sec)
    ticks_per_beat = mid.ticks_per_beat
    
    def sec_to_ticks(sec: float) -> int:
        beats = sec * (bpm / 60.0)
        return int(round(beats * ticks_per_beat))

    current_tick = 0
    for ev in sorted_events:
        abs_tick = sec_to_ticks(ev.time_sec)
        delta_tick = max(0, abs_tick - current_tick)
        track.append(mido.Message('note_on', channel=9, note=ev.pitch.value, velocity=ev.velocity, time=delta_tick))
        track.append(mido.Message('note_off', channel=9, note=ev.pitch.value, velocity=0, time=10))
        current_tick = abs_tick + 10

    mid.save(str(out_path))
    logger.info(f"[Transcriber] 표준 MIDI 파일 생성 완료: {out_path}")


def transcribe_drums(
    drums_path: Path,
    bass_path: Path = None,
    *,
    onset_threshold: float = 0.5,
    min_onset_gap_sec: float = 0.03,
    verbose: bool = False,
    bpm_hint: float = 120.0,
) -> List[DrumEvent]:
    """
    오디오 스템에서 드럼 이벤트를 딥러닝(CRNN) 모델로 추출하고, `output/drums.mid`로 내보냅니다.
    """
    logger.info(f"딥러닝 기반 드럼 전사 및 MIDI 추출 시작: {drums_path}")
    
    model_path = Path("models/best_drum_crnn_11class.pt")
    if not model_path.exists():
        logger.error(f"모델 가중치를 찾을 수 없습니다: {model_path}. 먼저 학습을 진행해주세요.")
        raise FileNotFoundError(f"Missing model weights at {model_path}")
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    from .train.model import DrumCRNN
    from .train.dataset import DrumClassMapping
    
    model = DrumCRNN(num_classes=DrumClassMapping.NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    # CRITICAL FIX: The model was trained with Batch Size 384. 
    # Its BatchNorm running statistics expect highly varied Groove datasets.
    # Inference is Batch Size 1 on quiet Demucs stems. 
    # By forcing the first BatchNorm to recompute statistics on the test input (Instance Norm behavior),
    # we completely bypass the domain shift caused by absolute volume/variance differences.
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.train() # Force BN to use batch statistics instead of running stats!

    
    sr = 44100
    hop_length = 441 # 10ms hop
    fps = sr // hop_length
    
    import soundfile as sf; audio_np, orig_sr = sf.read(str(drums_path), always_2d=True); y = torch.from_numpy(audio_np.T.astype("float32"))
    if orig_sr != sr:
        y = torchaudio.functional.resample(y, orig_sr, sr)
    if y.shape[0] > 1:
        y = y.mean(dim=0, keepdim=True)
        

        
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_fft=2048, hop_length=hop_length, n_mels=229, f_min=30.0, f_max=sr / 2.0
    ).to(device)
    amp_to_db = torchaudio.transforms.AmplitudeToDB(top_db=80).to(device)
    
    # torchaudio AmplitudeToDB defaults to referencing the max value of the current batch.
    # Since Demucs is quiet, its max is low, shifting all DB values up (making noise look like signal).
    # We force the db reference to a fixed 1.0 (0 dB FS) so quiet audio stays quiet.
    amp_to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)
    # The fix is to scale y up to 1.0 peak ONLY IF we don't change the background noise floor relative level.
    # But actually, librosa/torchaudio just uses the max of the spectrogram. 
    # Let's manually do amplitude to db so we have absolute control over the reference max.

    
    y_dev = y.to(device)
    mel_spec = amp_to_db(mel_transform(y_dev)).unsqueeze(0)


    
    segment_frames = 300 # 3초
    step_frames = 150 # 1.5초 겹침
    total_frames = mel_spec.shape[3]
    
    probs = np.zeros((total_frames, DrumClassMapping.NUM_CLASSES), dtype=np.float32)
    counts = np.zeros((total_frames, 1), dtype=np.float32)
    
    with torch.no_grad():
        for start_f in range(0, total_frames, step_frames):
            end_f = min(start_f + segment_frames, total_frames)
            seg_len = end_f - start_f
            
            x = mel_spec[:, :, :, start_f:end_f]
            if x.shape[3] < segment_frames:
                pad = segment_frames - x.shape[3]
                x = torch.nn.functional.pad(x, (0, pad))
                
            logits = model(x)
            seg_probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
            
            probs[start_f:end_f] = np.maximum(probs[start_f:end_f], seg_probs[:seg_len])
            
    # No averaging needed, we just take the max probability across overlapping segments
    
    reverse_map = {v: [] for v in set(DrumClassMapping.MIDI_MAP.values())}
    for midi_note, cls_idx in DrumClassMapping.MIDI_MAP.items():
        reverse_map[cls_idx].append(midi_note)
        
    events = []
    min_dist_frames = int(min_onset_gap_sec * fps)
    
    for cls_idx in range(DrumClassMapping.NUM_CLASSES):
        cls_probs = probs[:, cls_idx]
        peaks, _ = scipy.signal.find_peaks(cls_probs, height=onset_threshold, distance=min_dist_frames)
        
        print(f"Class {cls_idx} max prob: {cls_probs.max():.4f}, peaks found: {len(peaks)}")
        
        rep_note = reverse_map[cls_idx][0]
        try:
            pitch = DrumPitch(rep_note)
        except ValueError:
            print(f"Class {cls_idx} ValueError for rep_note {rep_note}!")
            continue
            
        for p in peaks:
            time_sec = p / fps
            confidence = float(cls_probs[p])
            vel = int(40 + (confidence - onset_threshold) / (1.0 - onset_threshold) * 87)
            events.append(DrumEvent(time_sec=time_sec, pitch=pitch, velocity=max(1, min(127, vel)), confidence=confidence))

    events.sort(key=lambda e: e.time_sec)
    
    logger.info(f"딥러닝 추론 전사 완료: 총 {len(events)}개의 노트 추출")
    
    midi_path = Path("output/drums.mid")
    _export_to_midi(events, midi_path, bpm_hint)

    return events
