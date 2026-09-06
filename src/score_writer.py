"""
src/score_writer.py
────────────────────
PAS(Percussive Arts Society) 국제 표준 기보법 기반 LilyPond PDF 생성기.

수정 사항:
  - Voice 2 쉼표 축약: 1박에 킥 없으면 r4 단일 표기 (r16 r16 r16 r16 금지)
  - 8비트 그루브 정규화: 인식된 타격 데이터를 8비트 표준 패턴으로 정규화
  - BPM 하프타임 보정: 감지된 BPM >= 110 이면 / 2 로 보정
  - 필인 마디: 16분음표 전체 타격 그대로 표기
"""
from __future__ import annotations

import logging
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .transcriber import DrumPitch
from .quantizer import QuantizedMeasure, GridEvent, SUBDIVISIONS

logger = logging.getLogger(__name__)

# ── PAS 표준 LilyPond 드럼 식별자 매핑 ──────────────────────────────────────
LY_DRUM_NAMES: Dict[DrumPitch, str] = {
    DrumPitch.BD:         "bd",
    DrumPitch.HH_FOOT:    "hhp",
    DrumPitch.SD:         "sn",
    DrumPitch.RIM:        "ss",
    DrumPitch.HH_CLOSED:  "hh",
    DrumPitch.HH_OPEN:    "hho",
    DrumPitch.CRASH:      "cymc",
    DrumPitch.RIDE:       "cymr",
    DrumPitch.RIDE_BELL:  "rb",
    DrumPitch.TOM_H:      "tomh",
    DrumPitch.TOM_M:      "tommh",
    DrumPitch.TOM_F:      "tomfl",
    DrumPitch.SPLASH:     "cyms",
    DrumPitch.CRASH_2:    "cymc", # LilyPond uses cymc for general crash
}

# ── 필인 판별용 피치 집합 ────────────────────────────────────────────────────
FILL_PITCHES = {DrumPitch.TOM_H, DrumPitch.TOM_M, DrumPitch.TOM_F, DrumPitch.CRASH, DrumPitch.CRASH_2, DrumPitch.SPLASH}


def _get_lilypond_executable() -> str:
    path = shutil.which("lilypond")
    if path:
        return path
    try:
        import lilypond as lp
        exe_name = "lilypond.exe" if platform.system() == "Windows" else "lilypond"
        bin_path = lp.executable(exe_name)
        if bin_path:
            return str(bin_path)
    except ImportError:
        pass
    return "lilypond"


# ═══════════════════════════════════════════════════════════════
#  8비트 그루브 표준화 렌더러
# ═══════════════════════════════════════════════════════════════

def _simplify_beat(slots: List[List[str]]) -> str:
    def to_chord(notes):
        if not notes: return "r"
        if len(notes) == 1: return notes[0]
        return "<" + " ".join(notes) + ">"
        
    s = [to_chord(x) for x in slots]
    a, b, c, d = s[0], s[1], s[2], s[3]
    mask = "".join(["1" if x != "r" else "0" for x in s])
    
    if mask == "0000": return "r4"
    if mask == "1000": return f"{a}4"
    if mask == "0100": return f"r16 {b}16 r8"
    if mask == "0010": return f"r8 {c}8"
    if mask == "0001": return f"r8. {d}16"
    if mask == "1100": return f"{a}16 {b}16 r8"
    if mask == "1010": return f"{a}8 {c}8"
    if mask == "1001": return f"{a}8. {d}16"
    if mask == "0110": return f"r16 {b}16 {c}8"
    if mask == "0101": return f"r16 {b}16 r16 {d}16"
    if mask == "0011": return f"r8 {c}16 {d}16"
    if mask == "1110": return f"{a}16 {b}16 {c}8"
    if mask == "1101": return f"{a}16 {b}16 r16 {d}16"
    if mask == "1011": return f"{a}8 {c}16 {d}16"
    if mask == "0111": return f"r16 {b}16 {c}16 {d}16"
    if mask == "1111": return f"{a}16 {b}16 {c}16 {d}16"
    return "r4"

def _render_voice(grid: Dict[int, List[GridEvent]], voice_idx: int) -> str:
    parts = []
    for beat in range(4):
        slots = []
        for slot in range(beat * 4, beat * 4 + 4):
            evs = [e for e in grid.get(slot, []) if e.voice == voice_idx]
            # Map pitch to lilypond name and eliminate duplicates
            notes = list(set(LY_DRUM_NAMES.get(e.pitch, "sn") for e in evs))
            notes.sort() # Ensure consistent ordering inside chords
            slots.append(notes)
        parts.append(_simplify_beat(slots))
    
    prefix = "\\stemUp " if voice_idx == 1 else "\\stemDown "
    return prefix + " ".join(parts)

def _build_ly_measure(m: QuantizedMeasure, is_fill: bool = False) -> str:
    v1 = _render_voice(m.grid, 1)
    v2 = _render_voice(m.grid, 2)
    return f"    << {{ {v1} }} \\\\ {{ {v2} }} >>"

def _measures_are_similar(a: QuantizedMeasure, b: QuantizedMeasure) -> bool:
    """두 마디가 동일한 킥/스네어 패턴인지 비교합니다."""
    KEY_PITCHES = {DrumPitch.BD, DrumPitch.SD, DrumPitch.HH_CLOSED, DrumPitch.HH_OPEN}
    for pos in range(SUBDIVISIONS):
        a_p = {e.pitch for e in a.grid.get(pos, []) if e.pitch in KEY_PITCHES}
        b_p = {e.pitch for e in b.grid.get(pos, []) if e.pitch in KEY_PITCHES}
        if a_p != b_p:
            return False
    return True


def _groove_desc(m: QuantizedMeasure) -> str:
    pitches = {e.pitch for e in m.events}
    if DrumPitch.RIDE in pitches:
        return "라이드(Ride) 중심 8비트 그루브"
    if DrumPitch.HH_OPEN in pitches:
        return "오픈 하이햇(Open HH) 강렬한 리듬"
    kick_count = sum(1 for e in m.events if e.pitch == DrumPitch.BD)
    if kick_count >= 4:
        return "16비트 당김음(Syncopation) 킥 컴핑"
    return "8비트 클로즈드 하이햇 기본 그루브"


def _fill_desc(m: QuantizedMeasure) -> str:
    if not m.events:
        return "해당 없음"
    pitches = {e.pitch for e in m.events}
    parts = []
    if pitches & {DrumPitch.TOM_H, DrumPitch.TOM_M, DrumPitch.TOM_F}:
        parts.append("Toms 런")
    if DrumPitch.CRASH in pitches:
        parts.append("Crash 액센트")
    if DrumPitch.SD in pitches:
        parts.append("Snare 액센트")
    return " + ".join(parts) if parts else "16분음표 필인"


def _correct_bpm(raw_bpm: float) -> float:
    """
    BPM 하프타임 보정: 워십 발라드에서 8분음표를 4분음표로 오인할 경우
    BPM >= 110 이면 절반으로 줄입니다.
    """
    if raw_bpm >= 110:
        return raw_bpm / 2.0
    return raw_bpm


def _generate_ly_source(
    measures: List[QuantizedMeasure],
    title: str,
    composer: str,
    bpm: float,
) -> str:
    # BPM 하프타임 보정
    corrected_bpm = _correct_bpm(bpm)
    bpm_int = int(round(corrected_bpm))

    header = f'''\\version "2.24.0"
\\header {{
  title = "{title}"
  subtitle = "CCM Drum Roadmap — PAS Standard  (BPM: {bpm_int})"
  composer = "{composer}"
  tagline = ""
}}
\\paper {{
  #(set-paper-size "a4")
  system-system-spacing.basic-distance = #18
  score-markup-spacing.basic-distance = #3
  markup-system-spacing.basic-distance = #5
  top-margin = #12
  bottom-margin = #12
  left-margin = #15
  right-margin = #15
  print-page-number = ##t
  max-systems-per-page = 4
}}
'''

    # 섹션 분리
    sections: List[Tuple[str, List[QuantizedMeasure]]] = []
    current_label = "Intro"
    current_section: List[QuantizedMeasure] = []

    for m in measures:
        if m.section_label:
            if current_section:
                sections.append((current_label, current_section))
            current_label = m.section_label
            current_section = [m]
        else:
            current_section.append(m)
    if current_section:
        sections.append((current_label, current_section))

    body_lines: List[str] = []

    for label, sec_measures in sections:
        if not sec_measures:
            continue

        groove_m = sec_measures[0]

        # 필인 마디: is_fill 플래그 우선, 없으면 마지막 마디
        fill_m = sec_measures[-1]
        for m in reversed(sec_measures):
            if m.is_fill and m.events:
                fill_m = m
                break

        # 반복 횟수 계산
        similar_count = sum(1 for m in sec_measures if _measures_are_similar(m, groove_m))
        repeat_count = max(0, similar_count - 1)

        groove_ly = _build_ly_measure(groove_m, is_fill=False)
        fill_ly   = _build_ly_measure(fill_m,   is_fill=True)

        if repeat_count >= 2:
            repeat_block = f"    \\repeat percent {repeat_count} {{\n{groove_ly}\n    }}"
        elif repeat_count == 1:
            repeat_block = f"{groove_ly}\n    \\repeat percent 1 {{\n{groove_ly}\n    }}"
        else:
            repeat_block = groove_ly

        gd = _groove_desc(groove_m)
        fd = _fill_desc(fill_m)

        score_block = f'''
\\score {{
  \\new DrumStaff \\with {{
    drumStyleTable = #percussion-style
    \\numericTimeSignature
    fontSize = #-1
    \\override StaffSymbol.staff-space = #(magstep -1)
  }} <<
    \\new DrumVoice {{
      \\voiceOne
      \\drummode {{
        \\tempo 4 = {bpm_int}
        \\time 4/4
        \\mark \\markup {{ \\bold \\box \\large "{label}" }}
{repeat_block}
        \\mark \\markup {{ \\italic \\small "Fill" }}
{fill_ly}
      }}
    }}
  >>
  \\layout {{
    indent = 2\\cm
    \\context {{
      \\DrumStaff
      \\consists "Merge_rests_engraver"
      \\override BarLine.hair-thickness = #3.5
    }}
  }}
}}
\\markup {{
  \\vspace #0.5
  \\fill-line {{
    \\rounded-box \\pad-markup #1.5 \\column {{
      \\line {{ \\bold "[구간 리듬]  " "{gd}" }}
      \\line {{ \\bold "[줄별 필인]  " "{fd}" }}
    }}
  }}
}}
\\markup {{ \\vspace #1.5 }}
'''
        body_lines.append(score_block)

    return header + "\n".join(body_lines)


def write_score(
    measures: List[QuantizedMeasure],
    output_stem: Path,
    *,
    title: str = "Drum Score",
    composer: str = "CCM Drum Transcriber",
    bpm: float = 120.0,
    verbose: bool = False,
) -> Path:
    ly_source = _generate_ly_source(measures, title, composer, bpm)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    ly_path  = output_stem.with_suffix(".ly")
    pdf_path = output_stem.with_suffix(".pdf")

    ly_path.write_text(ly_source, encoding="utf-8")
    logger.info(f"LilyPond 소스 생성 완료: {ly_path}")

    lily_exe = _get_lilypond_executable()
    try:
        result = subprocess.run(
            [lily_exe, "-o", str(output_stem.resolve()), str(ly_path.resolve())],
            check=True, capture_output=True, text=True, encoding="utf-8"
        )
        if verbose and result.stdout:
            logger.debug(result.stdout)
        logger.info(f"PDF 컴파일 완료: {pdf_path}")
        return pdf_path
    except subprocess.CalledProcessError as e:
        logger.error(f"LilyPond 컴파일 실패:\n{e.stderr}")
        raise RuntimeError("LilyPond 컴파일에 실패했습니다.") from e
    except FileNotFoundError:
        raise RuntimeError(
            "lilypond 실행 파일을 찾을 수 없습니다. "
            "`pip install lilypond` 또는 lilypond.org에서 설치하세요."
        )
