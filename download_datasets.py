"""
download_datasets.py

학습에 필요한 데이터셋 압축 해제 스크립트.

FSD50K 파일 위치: data/ 루트에 바로 놓으면 됨
  data/FSD50K.dev_audio.zip + .z01~.z05
  data/FSD50K.ground_truth.zip

Groove 오디오 위치:
  data/groove-v1.0.0.zip

사용법:
  python download_datasets.py --fsd50k-extract        # FSD50K 압축해제
  python download_datasets.py --groove-audio-extract  # Groove 압축해제
  python download_datasets.py --check                 # 데이터셋 상태 확인
"""

import argparse
import logging
import re
import subprocess
import sys
import zipfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("./data")


# ---------------------------------------------------------------------------
# FSD50K
# ---------------------------------------------------------------------------

def extract_fsd50k():
    """FSD50K dev audio zip parts 합쳐서 압축해제. data/ 루트에 바로 추출."""
    main_zip = DATA_DIR / "FSD50K.dev_audio.zip"

    if not main_zip.exists():
        logger.error(f"Not found: {main_zip}")
        logger.error("data/ 폴더에 FSD50K.dev_audio.zip 과 .z01~.z05 파일을 넣어주세요.")
        logger.error("Download from: https://zenodo.org/record/4060432")
        return

    out_dir = DATA_DIR / "FSD50K.dev_audio"
    if out_dir.exists() and any(out_dir.iterdir()):
        logger.info(f"Already extracted: {out_dir}")
    else:
        logger.info(f"Extracting FSD50K dev audio -> {out_dir} ...")
        out_dir.mkdir(parents=True, exist_ok=True)

        # 7z 경로 탐색 (PATH에 없으면 기본 설치 경로 사용)
        import shutil
        sevenzip = shutil.which("7z") or r"C:\Program Files\7-Zip\7z.exe"

        try:
            result = subprocess.run(
                [sevenzip, "x", str(main_zip), f"-o{DATA_DIR}", "-y"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                logger.info("Extracted with 7z.")
            else:
                logger.error(f"7z failed: {result.stderr}")
                return
        except FileNotFoundError:
            logger.error(f"7z not found at {sevenzip}. Install: winget install 7zip.7zip")
            return

    # ground_truth 압축해제
    gt_zip = DATA_DIR / "FSD50K.ground_truth.zip"
    gt_dir = DATA_DIR / "FSD50K.ground_truth"
    if gt_zip.exists() and not gt_dir.exists():
        logger.info("Extracting FSD50K.ground_truth.zip ...")
        with zipfile.ZipFile(gt_zip) as z:
            z.extractall(DATA_DIR)
        logger.info(f"Extracted -> {gt_dir}")
    elif gt_dir.exists():
        logger.info(f"ground_truth already extracted: {gt_dir}")
    else:
        logger.warning(f"FSD50K.ground_truth.zip not found in {DATA_DIR}")


# ---------------------------------------------------------------------------
# Groove
# ---------------------------------------------------------------------------

def extract_groove_audio():
    audio_zip = DATA_DIR / "groove-v1.0.0.zip"
    if not audio_zip.exists():
        logger.error(f"Not found: {audio_zip}")
        logger.error("data/ 폴더에 groove-v1.0.0.zip 을 넣어주세요.")
        return

    out_dir = DATA_DIR / "groove"
    if out_dir.exists() and any(out_dir.iterdir()):
        logger.info(f"Already extracted: {out_dir}")
        return

    logger.info("Extracting Groove audio ...")

    def _safe_name(name: str) -> str:
        """Windows에서 유효하지 않은 제어문자 제거."""
        return re.sub(r'[\r\n\x00]', '', name)

    INVALID_CHARS = set('<>:"/\\|?*') | {chr(i) for i in range(32)}

    with zipfile.ZipFile(audio_zip) as z:
        members = z.infolist()
        total = len(members)
        for i, member in enumerate(members, 1):
            safe = _safe_name(member.filename)
            if safe != member.filename:
                logger.warning(f"  Skipping invalid filename: {repr(member.filename)}")
                continue
            if not safe.strip():
                continue
            basename = safe.replace('/', '\\').split('\\')[-1]
            if any(c in basename for c in INVALID_CHARS):
                logger.warning(f"  Skipping: {repr(safe)}")
                continue
            try:
                z.extract(member, DATA_DIR)
            except Exception as e:
                logger.warning(f"  Skipping {member.filename!r}: {e}")
                continue
            if i % 500 == 0 or i == total:
                print(f"\r  {i}/{total} files extracted...", end="", flush=True)
    print()
    logger.info("Done.")


# ---------------------------------------------------------------------------
# Status check
# ---------------------------------------------------------------------------

def check_datasets():
    """어떤 데이터셋이 사용 가능한지 보고."""
    print("\n=== Dataset Status ===")
    checks = {
        "FSD50K":  DATA_DIR / "FSD50K.ground_truth" / "dev.csv",
        "Groove":  DATA_DIR / "groove" / "info.csv",
    }
    for name, path in checks.items():
        status = "[OK]     " if path.exists() else "[MISSING]"
        print(f"  {status}  {name:10s} -> {path}")

    # FSD50K 오디오 파일 수 확인
    fsd_audio = DATA_DIR / "FSD50K.dev_audio"
    if fsd_audio.exists():
        n = sum(1 for _ in fsd_audio.rglob("*.wav"))
        print(f"             FSD50K audio: {n} wav files in {fsd_audio}")

    # Groove 오디오 파일 수
    groove_dir = DATA_DIR / "groove"
    if groove_dir.exists():
        n = sum(1 for _ in groove_dir.rglob("*.wav"))
        print(f"             Groove audio: {n} wav files in {groove_dir}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset extraction helper")
    parser.add_argument("--fsd50k-extract",       action="store_true", help="FSD50K 압축해제")
    parser.add_argument("--groove-audio-extract", action="store_true", help="Groove 오디오 압축해제")
    parser.add_argument("--check",                action="store_true", help="데이터셋 상태 확인")
    args = parser.parse_args()

    if args.check or not any(vars(args).values()):
        check_datasets()
        sys.exit(0)

    if args.fsd50k_extract:
        extract_fsd50k()
    if args.groove_audio_extract:
        extract_groove_audio()

    check_datasets()
