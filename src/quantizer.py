"""
src/quantizer.py
────────────────────
추출된 DrumEvent(MIDI 노트)를 마디/비트 그리드에 맞추어 양자화(Quantize)합니다.
16분음표 단위로 스냅하며, 동일 틱에 발생한 중복/고스트 노이즈를 필터링합니다.
로드맵 분석을 통해 반복되는 마디를 식별(is_simile)합니다.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from .transcriber import DrumEvent, DrumPitch
from .beat_tracker import TempoGrid

logger = logging.getLogger(__name__)

# 4/4 박자 기준, 한 마디를 16개의 16분음표 슬롯으로 분할
SUBDIVISIONS = 16  

@dataclass
class GridEvent:
    pitch: DrumPitch
    velocity: int
    voice: int       # 1: 손 (Hi-Hat, Snare, Cymbal, Tom), 2: 발 (Kick, HH Pedal)

@dataclass
class QuantizedMeasure:
    measure_idx: int
    events: List[GridEvent] = field(default_factory=list)
    section_label: Optional[str] = None
    is_simile: bool = False
    is_fill: bool = False
    
    # 0 ~ 15 인덱스의 16분음표 그리드
    grid: Dict[int, List[GridEvent]] = field(default_factory=dict)
    
    def add_event(self, grid_pos: int, event: GridEvent):
        if grid_pos not in self.grid:
            self.grid[grid_pos] = []
        # 동일 틱(grid_pos), 동일 pitch 중복 방지 (노이즈 필터링)
        if any(e.pitch == event.pitch for e in self.grid[grid_pos]):
            return
        self.grid[grid_pos].append(event)
        self.events.append(event)


def _get_voice_for_pitch(pitch: DrumPitch) -> int:
    """손 파트(Voice 1)와 발 파트(Voice 2) 분리"""
    if pitch in (DrumPitch.BD, DrumPitch.HH_FOOT):
        return 2
    return 1


def _compute_measure_hash(m: QuantizedMeasure) -> str:
    """마디의 리듬 형태를 문자열로 해싱하여 비교용으로 사용"""
    parts = []
    for pos in range(SUBDIVISIONS):
        if pos in m.grid:
            pitches = sorted([e.pitch.value for e in m.grid[pos]])
            parts.append(f"{pos}:{'-'.join(map(str, pitches))}")
    return "|".join(parts)


def quantize(
    events: List[DrumEvent],
    grid: TempoGrid,
    section_hints: List[str] = None,
) -> List[QuantizedMeasure]:
    """이벤트를 16분음표 그리드에 스냅하여 마디별로 분류합니다."""
    logger.info("양자화(Quantization) 및 로드맵 분석 시작")
    
    measures: List[QuantizedMeasure] = []
    for i in range(grid.measure_count):
        measures.append(QuantizedMeasure(measure_idx=i))
        
    for ev in events:
        # 이벤트가 어느 마디에 속하는지 탐색
        m_idx = 0
        for i, dt in enumerate(grid.downbeat_times):
            if ev.time_sec >= dt:
                m_idx = i
            else:
                break
                
        if m_idx >= grid.measure_count:
            continue
            
        m = measures[m_idx]
        start_time = grid.downbeat_times[m_idx]
        
        # 다음 마디의 시작 시간 (없으면 bpm으로 추정)
        if m_idx + 1 < grid.measure_count:
            end_time = grid.downbeat_times[m_idx + 1]
        else:
            end_time = start_time + (60.0 / grid.bpm) * grid.time_sig_num
            
        measure_duration = end_time - start_time
        sixteenth_duration = measure_duration / SUBDIVISIONS
        
        # 16분음표 그리드 인덱스 계산 (0 ~ 15)
        offset = ev.time_sec - start_time
        grid_pos = int(round(offset / sixteenth_duration))
        
        if grid_pos >= SUBDIVISIONS:
            # 다음 마디 첫 박으로 넘어가는 경우
            if m_idx + 1 < grid.measure_count:
                measures[m_idx + 1].add_event(0, GridEvent(ev.pitch, ev.velocity, _get_voice_for_pitch(ev.pitch)))
            continue
            
        m.add_event(grid_pos, GridEvent(ev.pitch, ev.velocity, _get_voice_for_pitch(ev.pitch)))
        
    # ── Simile(반복 축약) 및 Fill(필인) 판별 ──
    # 4마디, 8마디 등은 보통 Fill이거나 섹션 전환부입니다.
    for i, m in enumerate(measures):
        if i == 0 or m.section_label:
            continue
            
        prev_m = measures[i - 1]
        
        # 이전 마디와 리듬이 95% 동일하면 Simile(%) 처리
        if _compute_measure_hash(m) == _compute_measure_hash(prev_m):
            m.is_simile = True
            m.grid.clear()
            m.events.clear()
        else:
            # 리듬이 달라졌고, 마디 수가 4의 배수 주변이면 Fill로 간주
            if (i + 1) % 4 == 0:
                m.is_fill = True

    simile_count = sum(1 for m in measures if m.is_simile)
    logger.info(f"양자화 완료: 총 {len(measures)}마디 (Simile 축약: {simile_count}마디)")
    return measures
