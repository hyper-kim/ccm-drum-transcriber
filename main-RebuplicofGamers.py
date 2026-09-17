"""
main.py
CCM 드럼 자동 채보 파이프라인 CLI 진입점.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

console = Console()


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ccm-drum-transcriber",
        description="CCM 워십 드럼 자동 채보 파이프라인",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python main.py --url "https://youtu.be/XXXXXXXXX" --title "God of Wonders"
  python main.py --url "..." --output ./sheets --form "Intro-A-B-Bridge-C-Outro"
""",
    )
    parser.add_argument("--url", "-u", required=True, help="YouTube 영상 URL")
    parser.add_argument("--output", "-o", default="./output", help="출력 디렉토리")
    parser.add_argument("--title", "-t", default=None, help="악보 제목")
    parser.add_argument("--form", default=None, help="송폼 (예: Intro-Verse-Chorus-Outro)")
    parser.add_argument("--time-sig", default="4/4", help="박자표 (기본: 4/4)")
    parser.add_argument("--model", default="htdemucs", help="Demucs 모델명")
    parser.add_argument("--device", default="auto", help="cuda/cpu/auto")
    parser.add_argument("--keep-audio", action="store_true", help="임시 WAV 파일 유지")
    parser.add_argument("--skip-separate", action="store_true", help="Demucs 분리 건너뛰기")
    parser.add_argument("--drums-wav", default=None, help="기존 drums.wav 경로")
    parser.add_argument("--pdf", action="store_true", help="LilyPond PDF 악보도 생성")
    parser.add_argument("--verbose", "-v", action="store_true", help="상세 로그")
    return parser.parse_args()


def _parse_time_sig(ts_str: str) -> tuple[int, int]:
    try:
        num, den = ts_str.split("/")
        return int(num), int(den)
    except ValueError:
        console.print(f"[red]박자표 오류: '{ts_str}'. 예: 4/4[/red]")
        sys.exit(1)


def _parse_form(form_str: str | None) -> list[str] | None:
    if not form_str:
        return None
    return [s.strip() for s in form_str.split("-") if s.strip()]


def run_pipeline(args: argparse.Namespace) -> Path:
    from src.downloader import download_audio
    from src.separator import separate_drums as separate_stems, separate_drums_fallback
    from src.beat_tracker import track_tempo
    from src.transcriber import transcribe_drums
    from src.quantizer import quantize
    from src.text_writer import write_text_tab

    output_dir = Path(args.output)
    time_sig = _parse_time_sig(args.time_sig)
    section_hints = _parse_form(args.form)

    console.print(Panel.fit(
        "[bold cyan]CCM 드럼 자동 채보 파이프라인[/bold cyan]\n"
        f"URL: [yellow]{args.url}[/yellow]",
        border_style="cyan",
    ))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:

        # Step 1: 다운로드
        if not args.drums_wav:
            task = progress.add_task("[cyan]1/5 유튜브 다운로드 중...", total=None)
            try:
                dl_result = download_audio(args.url, output_dir / "audio", verbose=args.verbose)
                title = args.title or dl_result.title
                raw_wav = dl_result.path
                progress.update(task, description=f"[green]✓ 다운로드 완료: {dl_result.title}")
            except Exception as exc:
                progress.stop()
                console.print(f"[red]다운로드 실패: {exc}[/red]")
                sys.exit(1)
        else:
            title = args.title or "DrumTranscription"
            raw_wav = Path(args.drums_wav)
            progress.add_task("[green]✓ 기존 오디오 파일 사용", total=1)

        # Step 2: 멀티 스템 분리 (drums, bass, other)
        task = progress.add_task("[cyan]2/5 멀티 스템 분리 중 (Demucs)...", total=None)
        if args.drums_wav:
            stems = {"drums": Path(args.drums_wav), "bass": None, "other": None}
            progress.update(task, description="[green]✓ 기존 drums.wav 사용")
        elif args.skip_separate:
            stems = separate_drums_fallback(raw_wav, output_dir)
            progress.update(task, description="[yellow]✓ Demucs 건너뜀 (폴백)")
        else:
            try:
                stems = separate_stems(raw_wav, output_dir, model_name=args.model, device=args.device, verbose=args.verbose)
                progress.update(task, description="[green]✓ 멀티 스템 분리 완료")
            except Exception as exc:
                progress.update(task, description="[yellow]✓ Demucs 실패, 폴백")
                logging.warning(f"Demucs 실패: {exc}")
                stems = separate_drums_fallback(raw_wav, output_dir)

        drums_wav = stems.get("drums")
        bass_wav  = stems.get("bass")
        other_wav = stems.get("other")

        # Step 3: BPM / 다운비트 분석
        task = progress.add_task("[cyan]3/5 BPM & 마디 분석 중...", total=None)
        try:
            grid = track_tempo(
                drums_wav,
                bass_path=bass_wav,
                other_path=other_wav,
                time_sig_num=time_sig[0],
                time_sig_den=time_sig[1],
                verbose=args.verbose,
            )
            progress.update(task, description=f"[green]✓ BPM: {grid.bpm:.1f}, 마디: {grid.measure_count}")
        except Exception as exc:
            progress.stop()
            console.print(f"[red]마디 분석 실패: {exc}[/red]")
            sys.exit(1)

        # Step 4: 딥러닝 드럼 채음 (CRNN 추론)
        task = progress.add_task("[cyan]4/5 딥러닝 드럼 채음 중...", total=None)
        try:
            events = transcribe_drums(drums_wav, bpm_hint=grid.bpm)
            progress.update(task, description=f"[green]✓ DrumEvent {len(events)}개 감지")
        except FileNotFoundError as exc:
            progress.stop()
            console.print(f"[red]{exc}[/red]")
            console.print("[yellow]먼저 python -m src.train.train 으로 모델을 학습시켜 주세요.[/yellow]")
            sys.exit(1)
        except Exception as exc:
            progress.stop()
            console.print(f"[red]드럼 채음 실패: {exc}[/red]")
            sys.exit(1)

        # Step 5: 양자화 & 텍스트 탭 생성
        task = progress.add_task("[cyan]5/5 양자화 & 악보 생성 중...", total=None)
        try:
            q_measures = quantize(events, grid, section_hints=section_hints)
            # PDF 악보 생성 (옵션)
            if args.pdf:
                from src.score_writer import write_score
                try:
                    pdf_path = write_score(
                        q_measures,
                        output_dir / "score" / Path(title).name,
                        title=title, bpm=grid.bpm, verbose=args.verbose,
                    )
                    logging.info(f"PDF 악보 생성 완료: {pdf_path}")
                except Exception as pdf_exc:
                    logger.warning(f"PDF 생성 건너뜀: {pdf_exc}")

            output_path = write_text_tab(
                q_measures,
                output_dir / "score" / f"{Path(title).name}",
                title=title,
                bpm=grid.bpm,
            )
            progress.update(task, description="[green]✓ 텍스트 탭 악보 생성 완료")
        except Exception as exc:
            progress.stop()
            console.print(f"[red]악보 생성 실패: {exc}[/red]")
            sys.exit(1)

    simile_count = sum(1 for m in q_measures if m.is_simile)
    fill_count   = sum(1 for m in q_measures if m.is_fill)

    console.print(Panel(
        f"[bold green]✓ 완료![/bold green]\n\n"
        f"  제목    : {title}\n"
        f"  BPM     : {grid.bpm:.1f}\n"
        f"  총 마디 : {grid.measure_count}\n"
        f"  Simile  : {simile_count}마디 (% 처리)\n"
        f"  Fill    : {fill_count}마디 (구체적 표시)\n\n"
        f"  [cyan]출력 파일: {output_path}[/cyan]",
        title="파이프라인 결과",
        border_style="green",
    ))

    if not args.keep_audio and not args.drums_wav:
        try:
            import shutil
            audio_dir = output_dir / "audio"
            if audio_dir.exists():
                shutil.rmtree(audio_dir)
        except Exception:
            pass

    return output_path


def main() -> None:
    args = _parse_args()
    _setup_logging(args.verbose)
    run_pipeline(args)


if __name__ == "__main__":
    main()

