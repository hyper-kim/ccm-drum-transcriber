"""
src/separator.py
────────────────
Meta Demucs v4 (htdemucs) PyTorch API를 직접 호출하여
입력 WAV에서 drums 스템을 분리한다.

GPU 전략:
  - CUDA 감지 시 → GPU 자동 선택 (VRAM 가장 큰 카드)
  - GPU 1 (RTX 3090, 24GB) 등 비디오 출력 없는 카드를 연산 전용으로 우선 사용
  - shifts=4, overlap=0.5 로 최고 품질 (GPU 있을 때)
  - CPU 폴백 시 shifts=1, overlap=0.25 로 속도 우선

반환값:
    Dict[str, Path] → 분리된 스템들의 경로 (drums, bass, other)
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "htdemucs"
_SAMPLE_RATE = 44100


def _check_demucs_available() -> bool:
    try:
        import demucs  # noqa: F401
        return True
    except ImportError:
        return False


def _select_best_device(preferred: str = "auto") -> tuple[str, str]:
    """
    사용 가능한 최적 디바이스를 선택한다.

    Returns
    -------
    (device_str, description)
      device_str : 'cuda:0', 'cuda:1', 'cpu' 등
      description: 로그용 설명 문자열
    """
    if preferred not in ("auto", "cuda"):
        return preferred, f"사용자 지정: {preferred}"

    if not torch.cuda.is_available():
        return "cpu", "CUDA 불가 → CPU 폴백"

    n = torch.cuda.device_count()
    if n == 0:
        return "cpu", "GPU 없음 → CPU 폴백"

    # VRAM이 가장 큰 GPU 선택 (단, 디스플레이 전용 GPU 후순위)
    best_idx = 0
    best_mem = 0
    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        mem = props.total_memory
        logger.info(
            f"  GPU {i}: {props.name} | VRAM {mem / 1024**3:.1f} GB"
        )
        if mem > best_mem:
            best_mem = mem
            best_idx = i

    props = torch.cuda.get_device_properties(best_idx)
    desc = (
        f"GPU {best_idx}: {props.name} "
        f"({props.total_memory / 1024**3:.1f} GB VRAM)"
    )
    return f"cuda:{best_idx}", desc


def separate_drums(
    audio_path: Path,
    output_dir: Path,
    *,
    model_name: str = _DEFAULT_MODEL,
    device: str = "auto",
    verbose: bool = False,
) -> dict[str, Path]:
    """
    Demucs v4 API를 통해 drums 스템을 분리한다.

    GPU 있을 때 자동으로 최고 품질 설정 적용:
      shifts=4, overlap=0.5, segment=None (전체 메모리 활용)

    Parameters
    ----------
    audio_path  : 입력 WAV 파일 경로
    output_dir  : 출력 디렉토리
    model_name  : Demucs 모델 ('htdemucs', 'htdemucs_ft', 'mdx_extra')
    device      : 'auto' | 'cuda' | 'cuda:0' | 'cuda:1' | 'cpu'
    verbose     : 상세 로그 출력

    Returns
    -------
    Dict[str, Path] : 분리된 스템(drums, bass, other) 경로 딕셔너리

    Raises
    ------
    ImportError  : demucs 미설치 시
    RuntimeError : 분리 실패 시
    """
    if not _check_demucs_available():
        raise ImportError(
            "demucs가 설치되어 있지 않습니다. `pip install demucs` 를 실행하세요."
        )

    from demucs.api import Separator, save_audio

    # ── 디바이스 결정 ───────────────────────────────────────────────────────
    resolved_device, device_desc = _select_best_device(device)
    is_gpu = resolved_device.startswith("cuda")

    logger.info(f"[Demucs] 모델={model_name} | 디바이스={device_desc}")

    # ── GPU/CPU 품질 설정 ───────────────────────────────────────────────────
    if is_gpu:
        # GPU: 최고 품질 — shifts=4로 앙상블 (4배 더 정확)
        shifts   = 4
        overlap  = 0.5
        segment  = None   # 모델 기본값 사용 (VRAM 허용 범위에서 최대)
        logger.info(
            f"[Demucs] GPU 모드: shifts={shifts}, overlap={overlap} "
            f"(CPU 대비 ~4x 고품질)"
        )
    else:
        # CPU: 속도 우선
        shifts   = 1
        overlap  = 0.25
        segment  = None
        logger.warning(
            "[Demucs] CPU 모드: shifts=1 (CUDA PyTorch 설치 권장)"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    stems_dir = output_dir / "stems"
    stems_dir.mkdir(exist_ok=True)

    # ── Demucs Separator 초기화 ─────────────────────────────────────────────
    try:
        sep_kwargs: dict = dict(
            model=model_name,
            device=resolved_device,
            shifts=shifts,
            overlap=overlap,
            progress=verbose,
        )
        if segment is not None:
            sep_kwargs["segment"] = segment

        separator = Separator(**sep_kwargs)
    except Exception as exc:
        if is_gpu:
            logger.warning(f"GPU 초기화 실패 ({exc}), CPU로 재시도...")
            separator = Separator(
                model=model_name, device="cpu",
                shifts=1, overlap=0.25, progress=verbose,
            )
            resolved_device = "cpu"
        else:
            raise RuntimeError(f"Demucs 모델 로드 실패 ({model_name}): {exc}") from exc

    # ── 오디오 로드 및 분리 ─────────────────────────────────────────────────
    logger.info(f"[Demucs] 분리 시작: {audio_path.name}")
    try:
        origin, separated = separator.separate_audio_file(audio_path)
    except torch.cuda.OutOfMemoryError as oom:
        if is_gpu:
            logger.warning(
                f"GPU OOM 발생 ({oom}). segment를 줄여서 CPU로 폴백..."
            )
            torch.cuda.empty_cache()
            separator_cpu = Separator(
                model=model_name, device="cpu",
                shifts=1, overlap=0.25, progress=verbose,
            )
            origin, separated = separator_cpu.separate_audio_file(audio_path)
        else:
            raise RuntimeError(f"Demucs OOM: {oom}") from oom
    except Exception as exc:
        raise RuntimeError(f"Demucs 분리 실패: {exc}") from exc

    # ── 스템 저장 (drums, bass, other) ──────────────────────────────────────
    saved_paths = {}
    target_stems = ["drums", "bass", "other"]
    
    for stem_name in target_stems:
        if stem_name not in separated:
            logger.warning(f"모델 '{model_name}'이 '{stem_name}' 스템을 생성하지 않았습니다.")
            continue
            
        tensor = separated[stem_name]
        path = stems_dir / f"{audio_path.stem}_{stem_name}.wav"
        save_audio(tensor, path, samplerate=separator.samplerate)
        saved_paths[stem_name] = path
        logger.info(f"[Demucs] {stem_name} 스템 저장 완료: {path}")

    # GPU 메모리 명시적 해제
    if is_gpu:
        del origin, separated
        torch.cuda.empty_cache()
        logger.info("[Demucs] GPU 메모리 해제 완료")

    return saved_paths


def separate_drums_fallback(
    audio_path: Path,
    output_dir: Path,
    *,
    verbose: bool = False,
) -> dict[str, Path]:
    """
    Demucs 미설치 또는 실패 시 사용하는 단순 폴백.
    입력 WAV를 그대로 drums, bass, other.wav로 복사한다 (테스트/개발 목적).
    """
    import shutil

    output_dir.mkdir(parents=True, exist_ok=True)
    stems_dir = output_dir / "stems"
    stems_dir.mkdir(exist_ok=True)

    saved_paths = {}
    for stem_name in ["drums", "bass", "other"]:
        path = stems_dir / f"{audio_path.stem}_{stem_name}.wav"
        shutil.copy2(audio_path, path)
        saved_paths[stem_name] = path
        
    logger.warning("폴백: 원본 오디오를 drums, bass, other로 복사 (분리 안됨)")
    return saved_paths
