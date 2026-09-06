"""
src/text_writer.py
────────────────────
양자화된 드럼 이벤트를 직관적인 텍스트 형태(Drum Tab)로 출력합니다.
LilyPond 렌더링의 오류를 방지하고 추출된 데이터를 있는 그대로 보여줍니다.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from .transcriber import DrumPitch
from .quantizer import QuantizedMeasure, SUBDIVISIONS

logger = logging.getLogger(__name__)

def write_text_tab(
    measures: List[QuantizedMeasure],
    output_stem: Path,
    *,
    title: str = "Drum Score",
    bpm: float = 120.0,
) -> Path:
    """양자화된 마디 데이터를 Text Drum Tab 형태로 내보냅니다."""
    
    lines = []
    lines.append(f"Title: {title}")
    lines.append(f"BPM  : {bpm:.1f}")
    lines.append("=" * 40)
    lines.append("")
    
    # 탭에 표시할 주요 악기 그룹
    groups = {
        "CYM": [DrumPitch.CRASH, DrumPitch.RIDE, DrumPitch.RIDE_BELL],
        "HH ": [DrumPitch.HH_CLOSED, DrumPitch.HH_OPEN, DrumPitch.HH_FOOT],
        "TOM": [DrumPitch.TOM_H, DrumPitch.TOM_M, DrumPitch.TOM_F],
        "SNR": [DrumPitch.SD, DrumPitch.RIM],
        "KIK": [DrumPitch.BD],
    }
    
    for m in measures:
        if m.section_label:
            lines.append(f"\n[{m.section_label}]")
            
        header = f"Measure {m.measure_idx + 1:03d}"
        if m.is_simile:
            lines.append(f"{header} | % (Simile) |")
            continue
            
        if m.is_fill:
            header += " (Fill)"
            
        lines.append(header)
        
        # 16분음표 그리드(0~15)를 텍스트 탭으로 변환
        for group_name, pitches in groups.items():
            row_chars = []
            for pos in range(SUBDIVISIONS):
                # 가독성을 위해 4칸마다 구분선
                if pos > 0 and pos % 4 == 0:
                    row_chars.append("|")
                    
                hit = False
                if pos in m.grid:
                    for ev in m.grid[pos]:
                        if ev.pitch in pitches:
                            hit = True
                            break
                row_chars.append("x" if hit else "-")
                
            row_str = "".join(row_chars)
            # 해당 라인에 타격이 하나라도 있거나, 킥/스네어/하이햇은 기본으로 출력
            if "x" in row_str or group_name in ["HH ", "SNR", "KIK"]:
                lines.append(f"{group_name} |{row_str}|")
                
        lines.append("") # 마디 간 띄어쓰기

    out_path = output_stem.with_suffix(".txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    
    logger.info(f"텍스트 탭 악보 생성 완료: {out_path}")
    return out_path
