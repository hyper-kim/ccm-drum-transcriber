import re

with open("src/quantizer.py", "r", encoding="utf-8") as f:
    text = f.read()

old_str = '''    measures: List[QuantizedMeasure] = []
    for i in range(grid.measure_count):
        label = section_hints[i] if section_hints and i < len(section_hints) else None
        measures.append(QuantizedMeasure(measure_idx=i, section_label=label))'''

new_str = '''    measures: List[QuantizedMeasure] = []
    for i in range(grid.measure_count):
        measures.append(QuantizedMeasure(measure_idx=i))'''

old_str2 = '''        m.add_event(grid_pos, GridEvent(ev.pitch, ev.velocity, _get_voice_for_pitch(ev.pitch)))
        
    # 마디 Simile(반복 기호) 및 Fill(필인) 판별 로직'''

new_str2 = '''        m.add_event(grid_pos, GridEvent(ev.pitch, ev.velocity, _get_voice_for_pitch(ev.pitch)))
        
    # Song-form 지능형 섹션 분배 (Drum Energy 기반)
    if section_hints and len(section_hints) > 0:
        if len(section_hints) == 1:
            measures[0].section_label = section_hints[0]
        else:
            energies = []
            for m in measures:
                e = sum(ev.velocity for events_list in m.grid.values() for ev in events_list)
                energies.append(e)
            
            import numpy as np
            smoothed = np.convolve(energies, np.ones(4)/4.0, mode='same')
            diff = np.abs(np.diff(smoothed))
            for i in range(len(diff)):
                if (i + 1) % 4 == 0: diff[i] *= 1.2
                if (i + 1) % 8 == 0: diff[i] *= 1.5
            
            num_boundaries = len(section_hints) - 1
            boundaries = []
            sorted_indices = np.argsort(diff)[::-1]
            for idx in sorted_indices:
                idx += 1
                if idx < 4 or idx > len(measures) - 4: continue
                if all(abs(idx - b) >= 8 for b in boundaries):
                    boundaries.append(idx)
                if len(boundaries) >= num_boundaries:
                    break
            boundaries.sort()
            
            measures[0].section_label = section_hints[0]
            for i, b in enumerate(boundaries):
                if i + 1 < len(section_hints):
                    measures[b].section_label = section_hints[i + 1]
                    
    # 마디 Simile(반복 기호) 및 Fill(필인) 판별 로직'''

text = text.replace(old_str, new_str)
text = text.replace(old_str2, new_str2)

with open("src/quantizer.py", "w", encoding="utf-8") as f:
    f.write(text)

print("Patch applied successfully.")
