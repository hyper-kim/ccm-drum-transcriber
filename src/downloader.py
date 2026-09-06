"""
src/downloader.py
─────────────────
yt-dlp를 사용해 YouTube 영상에서 최고 품질 무손실 WAV(44100 Hz stereo)를 추출한다.

반환값:
    DownloadResult(path=Path, title=str, artist=str, duration_sec=float)
"""
from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yt_dlp


@dataclass
class DownloadResult:
    """다운로드 및 변환 결과"""
    path: Path            # 44.1kHz stereo WAV 경로
    title: str            # 트랙 제목
    artist: str           # 아티스트 / 채널명
    duration_sec: float   # 초 단위 길이
    url: str              # 원본 URL


def _sanitize(name: str) -> str:
    """파일시스템에 안전한 이름으로 변환"""
    return re.sub(r'[\\/*?:"<>|]', "_", name).strip()


def download_audio(
    url: str,
    output_dir: Path,
    *,
    sample_rate: int = 44100,
    channels: int = 2,
    verbose: bool = False,
) -> DownloadResult:
    """
    YouTube URL에서 오디오를 추출하여 WAV로 저장한다.

    Parameters
    ----------
    url         : YouTube 영상 URL
    output_dir  : WAV 파일을 저장할 디렉토리
    sample_rate : 목표 샘플레이트 (기본 44100 Hz)
    channels    : 채널 수 (기본 2, stereo)
    verbose     : yt-dlp 디버그 출력 여부

    Returns
    -------
    DownloadResult

    Raises
    ------
    RuntimeError : 다운로드 또는 변환 실패 시
    FileNotFoundError : ffmpeg 미설치 시
    """
    if shutil.which("ffmpeg") is None:
        raise FileNotFoundError(
            "ffmpeg가 설치되어 있지 않습니다. https://ffmpeg.org/download.html 에서 설치하세요."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: 메타데이터 추출 ─────────────────────────────────────────────
    meta: dict = {}
    ydl_opts_info: dict = {
        "quiet": not verbose,
        "no_warnings": not verbose,
        "skip_download": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts_info) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
            meta["title"] = info.get("title", "Unknown Title")
            meta["artist"] = info.get("uploader", info.get("channel", "Unknown Artist"))
            meta["duration"] = float(info.get("duration", 0.0))
        except Exception as exc:
            raise RuntimeError(f"메타데이터 추출 실패: {exc}") from exc

    safe_title = _sanitize(meta["title"])
    wav_path = output_dir / f"{safe_title}_raw.wav"

    # ── Step 2: 최고 품질 오디오 다운로드 + ffmpeg 변환 ─────────────────────
    ydl_opts_dl: dict = {
        "quiet": not verbose,
        "no_warnings": not verbose,
        "format": "bestaudio/best",
        "outtmpl": str(output_dir / f"{safe_title}.%(ext)s"),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "0",  # lossless
            }
        ],
        "postprocessor_args": [
            "-ar", str(sample_rate),
            "-ac", str(channels),
        ],
    }

    with yt_dlp.YoutubeDL(ydl_opts_dl) as ydl:
        try:
            ydl.download([url])
        except Exception as exc:
            raise RuntimeError(f"다운로드 실패: {exc}") from exc

    # yt-dlp가 실제로 저장한 파일 탐색
    candidates = list(output_dir.glob(f"{safe_title}*.wav"))
    if not candidates:
        raise RuntimeError(
            f"WAV 파일을 찾을 수 없습니다. output_dir={output_dir}, title={safe_title}"
        )

    actual_wav = max(candidates, key=lambda p: p.stat().st_size)

    # 파일명 정규화
    if actual_wav != wav_path:
        actual_wav.rename(wav_path)

    return DownloadResult(
        path=wav_path,
        title=meta["title"],
        artist=meta["artist"],
        duration_sec=meta["duration"],
        url=url,
    )
