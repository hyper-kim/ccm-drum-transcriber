"""
src/beat_tracker.py
────────────────────
librosa를 사용해 드럼 스템 오디오에서 BPM과
마디별 다운비트(Downbeat) 타이밍 그리드를 추출한다.

GPU 활용: torchaudio mel spectrogram을 CUDA에서 계산하여
         onset strength를 빠르게 추출 (GPU 있을 때 자동 사용).

반환값:
    TempoGrid(bpm, beat_times, downbeat_times, measure_count, time_sig_num, time_sig_den)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

import librosa
import numpy as np
import soundfile as sf
import torch

logger = logging.getLogger(__name__)


@dataclass
class TempoGrid:
    """템포 분석 결과"""
    bpm: float
    beat_times: np.ndarray
    downbeat_times: np.ndarray
    measure_count: int
    time_sig_num: int = 4
    time_sig_den: int = 4
    beat_duration_sec: float = 0.0
    sixteenth_duration_sec: float = 0.0

    def __post_init__(self) -> None:
        if self.bpm > 0:
            self.beat_duration_sec = 60.0 / self.bpm
            self.sixteenth_duration_sec = self.beat_duration_sec / 4.0


def _gpu_onset_strength(
    y: np.ndarray,
    sr: int,
    hop_length: int = 512,
    device: str = "cpu",
) -> np.ndarray:
    """
    torchaudio MelSpectrogram을 GPU에서 계산하여 onset strength를 추출.
    GPU가 없으면 librosa로 폴백.
    """
    if device == "cpu" or not torch.cuda.is_available():
        # librosa CPU 경로
        return librosa.onset.onset_strength(
            y=y, sr=sr,
            hop_length=hop_length,
            aggregate=np.median,
            fmax=8000,
        )

    try:
        import torchaudio
        import torchaudio.transforms as T

        y_tensor = torch.from_numpy(y.astype(np.float32)).unsqueeze(0).to(device)

        # GPU에서 Mel Spectrogram 계산
        mel_transform = T.MelSpectrogram(
            sample_rate=sr,
            n_fft=2048,
            hop_length=hop_length,
            n_mels=128,
            f_max=8000,
        ).to(device)

        with torch.no_grad():
            mel = mel_transform(y_tensor)          # (1, n_mels, T)
            mel_db = T.AmplitudeToDB()(mel)        # dB 스케일

        # onset strength: 연속 프레임 차이의 반파 정류
        mel_np = mel_db.squeeze(0).cpu().numpy()   # (n_mels, T)
        diff = np.diff(mel_np, axis=1)
        diff = np.maximum(diff, 0)                 # 반파 정류
        onset_env = np.mean(diff, axis=0)          # 채널 평균

        # librosa와 길이 맞추기 (1프레임 패딩)
        onset_env = np.concatenate([[0.0], onset_env])

        logger.debug(f"GPU onset strength 계산 완료 (device={device})")
        return onset_env

    except Exception as exc:
        logger.debug(f"GPU onset strength 실패 ({exc}), librosa CPU 폴백")
        return librosa.onset.onset_strength(
            y=y, sr=sr, hop_length=hop_length,
            aggregate=np.median, fmax=8000,
        )


def _estimate_bpm_robust(
    y: np.ndarray,
    sr: int,
    *,
    start_bpm: float = 120.0,
    device: str = "cpu",
) -> Tuple[float, np.ndarray]:
    """
    BPM과 비트 프레임을 추정한다.
    GPU에서 onset strength를 계산하여 속도를 높인다.
    """
    hop_length = 512
    onset_env = _gpu_onset_strength(y, sr, hop_length=hop_length, device=device)

    tempo, beats = librosa.beat.beat_track(
        onset_envelope=onset_env,
        sr=sr,
        start_bpm=start_bpm,
        tightness=100,
        trim=False,
    )

    bpm = float(np.atleast_1d(tempo)[0])
    while bpm < 60:
        bpm *= 2
    while bpm > 240:
        bpm /= 2

    return bpm, beats


def _align_downbeats(
    beat_frames: np.ndarray,
    y_other: np.ndarray,
    y_bass: np.ndarray,
    sr: int,
    hop_length: int,
    beats_per_measure: int,
) -> np.ndarray:
    """화성(other) 및 베이스(bass) 스템을 활용하여 다운비트(1박) 위상을 고정한다."""
    if y_other is None and y_bass is None:
        beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
        return beat_times[::beats_per_measure]
        
    y_harm = np.zeros_like(y_other) if y_other is not None else np.zeros_like(y_bass)
    if y_other is not None: y_harm += y_other
    if y_bass is not None: y_harm += y_bass
    
    onset_env = librosa.onset.onset_strength(y=y_harm, sr=sr, hop_length=hop_length)
    
    phase_energies = np.zeros(beats_per_measure)
    for i in range(beats_per_measure):
        phase_frames = beat_frames[i::beats_per_measure]
        valid_frames = [f for f in phase_frames if f < len(onset_env)]
        if valid_frames:
            phase_energies[i] = np.sum(onset_env[valid_frames])
            
    best_phase = int(np.argmax(phase_energies))
    downbeat_frames = beat_frames[best_phase::beats_per_measure]
    return librosa.frames_to_time(downbeat_frames, sr=sr, hop_length=hop_length)


def track_tempo(
    drums_path: Path,
    bass_path: Path = None,
    other_path: Path = None,
    *,
    time_sig_num: int = 4,
    time_sig_den: int = 4,
    start_bpm: float = 120.0,
    device: str = "auto",
    verbose: bool = False,
) -> TempoGrid:
    """
    드럼 WAV에서 BPM과 마디별 타이밍 그리드를 추출한다.

    Parameters
    ----------
    drums_path    : 드럼 스템 WAV 파일 경로
    time_sig_num  : 박자 분자 (기본 4)
    time_sig_den  : 박자 분모 (기본 4)
    start_bpm     : BPM 추정 시작점 힌트
    device        : 'auto' | 'cuda' | 'cuda:N' | 'cpu'
    verbose       : 상세 로그
    """
    # 디바이스 결정
    if device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        resolved_device = device

    logger.info(f"템포 분석 시작: {drums_path} (device={resolved_device})")

    y, sr = librosa.load(str(drums_path), sr=None, mono=True)
    logger.info(f"드럼 오디오 로드: sr={sr}Hz, duration={len(y)/sr:.1f}s")
    
    y_bass, y_other = None, None
    if bass_path and bass_path.exists():
        y_bass, _ = librosa.load(str(bass_path), sr=sr, mono=True)
    if other_path and other_path.exists():
        y_other, _ = librosa.load(str(other_path), sr=sr, mono=True)

    hop_length = 512
    bpm, beat_frames = _estimate_bpm_robust(
        y, sr, start_bpm=start_bpm, device=resolved_device
    )
    logger.info(f"추정 BPM: {bpm:.2f}")

    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
    
    logger.info("합주 컨텍스트(other, bass) 기반 다운비트(1박) 위상 락온(Lock-on) 계산 중...")
    downbeat_times = _align_downbeats(
        beat_frames, y_other, y_bass, sr, hop_length, beats_per_measure=time_sig_num
    )

    measure_count = len(downbeat_times)
    logger.info(f"합주 인지 마디 수: {measure_count}, 비트 수: {len(beat_times)}")

    return TempoGrid(
        bpm=bpm,
        beat_times=beat_times,
        downbeat_times=downbeat_times,
        measure_count=measure_count,
        time_sig_num=time_sig_num,
        time_sig_den=time_sig_den,
    )
