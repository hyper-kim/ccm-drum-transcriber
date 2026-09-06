"""
tests/test_pipeline.py
───────────────────────
4마디짜리 합성 드럼 WAV를 생성하여 파이프라인 전 과정을 검증한다.

테스트 전략:
  - 실제 유튜브 다운로드 / Demucs 분리는 건너뜀 (외부 의존성)
  - numpy로 합성 드럼 패턴(Kick, Snare, HH 루프) WAV를 직접 생성
  - beat_tracker → transcriber → quantizer → score_writer 순서로 통합 테스트
  - 최종 출력 파일(PDF 또는 XML) 존재 여부로 pass/fail 판단

합성 패턴 (BPM 120, 4/4):
  Kick  : 1박, 3박 (샘플 0, 2박)
  Snare : 2박, 4박 (샘플 1박, 3박)
  HH    : 매 16분음표 (8분음표 간격)
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


# ── 합성 드럼 WAV 생성 ────────────────────────────────────────────────────────

SR = 44100
BPM = 120.0
BEAT_SEC = 60.0 / BPM          # 0.5초
SIXTEENTH_SEC = BEAT_SEC / 4   # 0.125초
MEASURES = 4
BEATS_PER_MEASURE = 4
TOTAL_BEATS = MEASURES * BEATS_PER_MEASURE
TOTAL_SEC = TOTAL_BEATS * BEAT_SEC  # 8초


def _sine_burst(freq: float, duration_sec: float, sr: int, amplitude: float = 0.8) -> np.ndarray:
    """지정 주파수의 짧은 사인파 버스트를 생성한다."""
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    burst = amplitude * np.sin(2 * np.pi * freq * t)
    # 어택/릴리즈 엔벨로프
    env = np.exp(-t * 20)  # 빠른 감쇠
    return burst * env


def _noise_burst(duration_sec: float, sr: int, amplitude: float = 0.5) -> np.ndarray:
    """하이햇 시뮬: 고주파 노이즈 버스트"""
    n = int(sr * duration_sec)
    noise = amplitude * np.random.randn(n)
    t = np.linspace(0, duration_sec, n, endpoint=False)
    env = np.exp(-t * 80)  # 매우 빠른 감쇠
    # 고주파 필터링 시뮬 (간단히 odd harmonics 강조)
    return noise * env


def generate_synthetic_drums(
    output_path: Path,
    *,
    bpm: float = BPM,
    measures: int = MEASURES,
    sr: int = SR,
    include_fill: bool = True,
) -> Path:
    """
    합성 드럼 패턴 WAV를 생성한다.

    패턴:
      마디 1-3: 기본 그루브 (Kick: 1·3박, Snare: 2·4박, HH: 8분음표)
      마디 4  : 필인 (Tom 롤 + 크래시)
    """
    beat_sec = 60.0 / bpm
    total_sec = measures * BEATS_PER_MEASURE * beat_sec
    total_samples = int(sr * total_sec)

    y = np.zeros(total_samples, dtype=np.float32)

    kick_freq   = 60.0    # Hz — 저주파 킥
    snare_freq  = 220.0   # Hz — 스네어 (중간)
    tom_h_freq  = 180.0   # Hz — 하이탐
    tom_m_freq  = 130.0   # Hz — 미드탐
    tom_f_freq  = 90.0    # Hz — 플로어탐

    burst_dur = 0.05  # 50ms 버스트

    def add_at(time_sec: float, burst: np.ndarray) -> None:
        start = int(time_sec * sr)
        end = min(total_samples, start + len(burst))
        y[start:end] += burst[:end - start]

    for m in range(measures):
        m_start = m * BEATS_PER_MEASURE * beat_sec
        is_fill_measure = include_fill and (m == measures - 1)

        if not is_fill_measure:
            # ── 기본 그루브 ────────────────────────────────────────────────
            for beat in range(BEATS_PER_MEASURE):
                beat_t = m_start + beat * beat_sec

                # Hi-Hat: 매 8분음표 (beat + beat/2)
                for hh_sub in range(2):
                    hh_t = beat_t + hh_sub * (beat_sec / 2)
                    add_at(hh_t, _noise_burst(burst_dur, sr, 0.3))

                # Kick: 1박, 3박 (beat 0, 2)
                if beat in (0, 2):
                    add_at(beat_t, _sine_burst(kick_freq, burst_dur, sr, 0.9))

                # Snare: 2박, 4박 (beat 1, 3)
                if beat in (1, 3):
                    snare = (
                        _sine_burst(snare_freq, burst_dur, sr, 0.7)
                        + _noise_burst(burst_dur, sr, 0.4)
                    )
                    add_at(beat_t, snare)

        else:
            # ── 필인 마디 (마디 4) ─────────────────────────────────────────
            # 16분음표 간격으로 Tom 롤
            sixteenth = beat_sec / 4
            tom_seq = [tom_h_freq, tom_h_freq, tom_m_freq, tom_m_freq,
                       tom_f_freq, tom_f_freq, tom_f_freq, tom_f_freq]
            for i, freq in enumerate(tom_seq):
                t = m_start + i * sixteenth * 2  # 8분음표 간격
                add_at(t, _sine_burst(freq, burst_dur, sr, 0.8))

            # 크래시 마디 끝 (노이즈 + 저음)
            crash_t = m_start + (BEATS_PER_MEASURE - 1) * beat_sec
            crash = _noise_burst(0.3, sr, 0.6) + _sine_burst(180, 0.3, sr, 0.3)
            add_at(crash_t, crash)

    # 클리핑 방지 정규화
    max_val = np.max(np.abs(y))
    if max_val > 0:
        y = y / max_val * 0.9

    # Stereo 변환
    y_stereo = np.stack([y, y], axis=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), y_stereo, sr, subtype="PCM_16")
    return output_path


# ── 테스트 ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def synthetic_drums_wav(tmp_path_factory) -> Path:
    """세션 범위 합성 드럼 WAV 픽스처"""
    out_dir = tmp_path_factory.mktemp("drums")
    wav_path = out_dir / "synthetic_drums.wav"
    generate_synthetic_drums(wav_path)
    return wav_path


@pytest.fixture(scope="session")
def output_dir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("output")


# ──────────────────────────────────────────────────────────────────────────────
# 단위 테스트
# ──────────────────────────────────────────────────────────────────────────────

class TestSyntheticWavGeneration:
    """합성 WAV 생성 테스트"""

    def test_wav_file_created(self, synthetic_drums_wav: Path) -> None:
        assert synthetic_drums_wav.exists(), "합성 WAV 파일이 생성되지 않았습니다."

    def test_wav_duration(self, synthetic_drums_wav: Path) -> None:
        import soundfile as sf
        data, sr = sf.read(str(synthetic_drums_wav))
        duration = len(data) / sr
        expected = MEASURES * BEATS_PER_MEASURE * (60.0 / BPM)
        assert abs(duration - expected) < 0.1, (
            f"WAV 길이 오류: {duration:.2f}s (예상: {expected:.2f}s)"
        )

    def test_wav_stereo(self, synthetic_drums_wav: Path) -> None:
        import soundfile as sf
        data, sr = sf.read(str(synthetic_drums_wav))
        assert data.ndim == 2 and data.shape[1] == 2, "WAV가 스테레오가 아닙니다."

    def test_wav_sample_rate(self, synthetic_drums_wav: Path) -> None:
        import soundfile as sf
        data, sr = sf.read(str(synthetic_drums_wav))
        assert sr == SR, f"샘플레이트 오류: {sr} (예상: {SR})"


class TestBeatTracker:
    """beat_tracker 단위 테스트"""

    def test_returns_tempo_grid(self, synthetic_drums_wav: Path) -> None:
        from src.beat_tracker import track_tempo, TempoGrid
        grid = track_tempo(synthetic_drums_wav)
        assert isinstance(grid, TempoGrid)

    def test_bpm_in_range(self, synthetic_drums_wav: Path) -> None:
        from src.beat_tracker import track_tempo
        grid = track_tempo(synthetic_drums_wav)
        assert 80 <= grid.bpm <= 160, f"BPM {grid.bpm:.1f}이 합리적 범위를 벗어났습니다."

    def test_measure_count_positive(self, synthetic_drums_wav: Path) -> None:
        from src.beat_tracker import track_tempo
        grid = track_tempo(synthetic_drums_wav)
        assert grid.measure_count >= 1, "마디 수가 0 이하입니다."

    def test_downbeat_times_sorted(self, synthetic_drums_wav: Path) -> None:
        from src.beat_tracker import track_tempo
        import numpy as np
        grid = track_tempo(synthetic_drums_wav)
        assert np.all(np.diff(grid.downbeat_times) > 0), "다운비트 타임스탬프가 오름차순이 아닙니다."


class TestTranscriber:
    """transcriber 단위 테스트"""

    def test_returns_drum_events(self, synthetic_drums_wav: Path) -> None:
        from src.transcriber import transcribe_drums, DrumEvent
        events = transcribe_drums(synthetic_drums_wav)
        assert isinstance(events, list)
        assert len(events) > 0, "이벤트가 감지되지 않았습니다."

    def test_events_are_drum_events(self, synthetic_drums_wav: Path) -> None:
        from src.transcriber import transcribe_drums, DrumEvent
        events = transcribe_drums(synthetic_drums_wav)
        for ev in events[:5]:
            assert isinstance(ev, DrumEvent)

    def test_events_time_sorted(self, synthetic_drums_wav: Path) -> None:
        from src.transcriber import transcribe_drums
        events = transcribe_drums(synthetic_drums_wav)
        times = [e.time_sec for e in events]
        assert times == sorted(times), "DrumEvent가 시간 순서대로 정렬되지 않았습니다."

    def test_velocity_range(self, synthetic_drums_wav: Path) -> None:
        from src.transcriber import transcribe_drums
        events = transcribe_drums(synthetic_drums_wav)
        for ev in events:
            assert 1 <= ev.velocity <= 127, f"벨로시티 범위 오류: {ev.velocity}"

    def test_pitch_variety(self, synthetic_drums_wav: Path) -> None:
        """합성 패턴이므로 최소 2가지 이상 악기가 감지되어야 함"""
        from src.transcriber import transcribe_drums
        events = transcribe_drums(synthetic_drums_wav)
        pitches = set(ev.pitch for ev in events)
        assert len(pitches) >= 2, f"감지된 악기 종류가 너무 적음: {pitches}"


class TestQuantizer:
    """quantizer 단위 테스트"""

    def test_returns_measures(self, synthetic_drums_wav: Path) -> None:
        from src.beat_tracker import track_tempo
        from src.transcriber import transcribe_drums
        from src.quantizer import quantize, QuantizedMeasure

        grid = track_tempo(synthetic_drums_wav)
        events = transcribe_drums(synthetic_drums_wav)
        measures = quantize(events, grid)

        assert isinstance(measures, list)
        assert len(measures) >= 1

    def test_grid_positions_valid(self, synthetic_drums_wav: Path) -> None:
        from src.beat_tracker import track_tempo
        from src.transcriber import transcribe_drums
        from src.quantizer import quantize, SUBDIVISIONS

        grid = track_tempo(synthetic_drums_wav)
        events = transcribe_drums(synthetic_drums_wav)
        measures = quantize(events, grid)

        for m in measures:
            for ev in m.events:
                assert 0 <= ev.grid_position < SUBDIVISIONS, (
                    f"그리드 위치 범위 오류: {ev.grid_position}"
                )

    def test_voice_assignment(self, synthetic_drums_wav: Path) -> None:
        from src.beat_tracker import track_tempo
        from src.transcriber import transcribe_drums
        from src.quantizer import quantize
        from src.transcriber import DrumPitch

        grid = track_tempo(synthetic_drums_wav)
        events = transcribe_drums(synthetic_drums_wav)
        measures = quantize(events, grid)

        for m in measures:
            for ev in m.events:
                assert ev.voice in (1, 2), f"잘못된 Voice: {ev.voice}"

    def test_simile_not_on_first_measure(self, synthetic_drums_wav: Path) -> None:
        from src.beat_tracker import track_tempo
        from src.transcriber import transcribe_drums
        from src.quantizer import quantize

        grid = track_tempo(synthetic_drums_wav)
        events = transcribe_drums(synthetic_drums_wav)
        measures = quantize(events, grid)

        if measures:
            assert not measures[0].is_simile, "첫 번째 마디는 simile이면 안 됩니다."


# ──────────────────────────────────────────────────────────────────────────────
# 통합 테스트: 악보 생성
# ──────────────────────────────────────────────────────────────────────────────

class TestScoreWriter:
    """score_writer 단위 테스트"""

    def test_output_file_created(
        self,
        synthetic_drums_wav: Path,
        output_dir: Path,
    ) -> None:
        from src.beat_tracker import track_tempo
        from src.transcriber import transcribe_drums
        from src.quantizer import quantize
        from src.score_writer import write_score

        grid = track_tempo(synthetic_drums_wav)
        events = transcribe_drums(synthetic_drums_wav)
        measures = quantize(events, grid, section_hints=["Intro", "Groove", "Fill", "Outro"])

        out_path = write_score(
            measures,
            output_dir,
            title="TestDrumRoadmap",
            bpm=grid.bpm,
        )

        assert out_path.exists(), f"악보 파일이 생성되지 않았습니다: {out_path}"
        assert out_path.suffix in (".pdf", ".xml"), (
            f"예상치 못한 확장자: {out_path.suffix}"
        )
        assert out_path.stat().st_size > 100, "생성된 파일이 너무 작습니다 (빈 파일 가능성)."

    def test_output_with_section_hints(
        self,
        synthetic_drums_wav: Path,
        output_dir: Path,
    ) -> None:
        from src.beat_tracker import track_tempo
        from src.transcriber import transcribe_drums
        from src.quantizer import quantize
        from src.score_writer import write_score

        grid = track_tempo(synthetic_drums_wav)
        events = transcribe_drums(synthetic_drums_wav)
        measures = quantize(
            events, grid,
            section_hints=["Intro", "A", "B", "Outro"],
        )

        out_path = write_score(
            measures,
            output_dir / "sections",
            title="SectionTest",
            bpm=grid.bpm,
        )
        assert out_path.exists()


# ──────────────────────────────────────────────────────────────────────────────
# 파이프라인 통합 테스트
# ──────────────────────────────────────────────────────────────────────────────

class TestFullPipeline:
    """전체 파이프라인 통합 테스트 (Demucs/다운로드 제외)"""

    @pytest.mark.timeout(60)
    def test_end_to_end(self, synthetic_drums_wav: Path, output_dir: Path) -> None:
        """
        합성 드럼 WAV → beat_tracker → transcriber → quantizer → score_writer
        전 과정이 에러 없이 완료되고 출력 파일이 존재해야 한다.
        """
        from src.beat_tracker import track_tempo
        from src.transcriber import transcribe_drums
        from src.quantizer import quantize
        from src.score_writer import write_score

        # 1. 템포 분석
        grid = track_tempo(synthetic_drums_wav)
        assert grid.bpm > 0

        # 2. 전사
        events = transcribe_drums(synthetic_drums_wav)
        assert len(events) > 0

        # 3. 양자화
        measures = quantize(
            events, grid,
            section_hints=["Intro", "Verse", "Fill", "Outro"],
            simile_threshold=0.85,
        )
        assert len(measures) > 0

        # 4. 악보 생성
        out_path = write_score(
            measures,
            output_dir / "e2e",
            title="E2E_Test_Roadmap",
            bpm=grid.bpm,
        )

        assert out_path.exists(), f"최종 출력 파일 없음: {out_path}"
        file_size = out_path.stat().st_size
        assert file_size > 200, f"출력 파일이 너무 작음: {file_size} bytes"

        print(f"\n✓ 파이프라인 완료: {out_path} ({file_size} bytes)")

    def test_simile_reduces_output(self, synthetic_drums_wav: Path, output_dir: Path) -> None:
        """simile 처리가 실제로 마디를 축약하는지 검증"""
        from src.beat_tracker import track_tempo
        from src.transcriber import transcribe_drums
        from src.quantizer import quantize

        grid = track_tempo(synthetic_drums_wav)
        events = transcribe_drums(synthetic_drums_wav)

        # 높은 임계값 → simile 마디 증가
        measures_strict = quantize(events, grid, simile_threshold=0.50)
        simile_strict = sum(1 for m in measures_strict if m.is_simile)

        # 낮은 임계값 → simile 마디 감소
        measures_loose = quantize(events, grid, simile_threshold=0.99)
        simile_loose = sum(1 for m in measures_loose if m.is_simile)

        # 임계값이 낮을수록 simile이 더 많아야 함
        assert simile_strict >= simile_loose, (
            f"simile 감지 로직 오류: strict={simile_strict}, loose={simile_loose}"
        )
