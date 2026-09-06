from src.quantizer import quantize, TempoGrid, DrumEvent
from src.score_writer import _generate_ly_source
from src.transcriber import DrumPitch

events = [
    DrumEvent(time_sec=0.0, pitch=DrumPitch.BD, velocity=100, confidence=0.9),
    DrumEvent(time_sec=0.5, pitch=DrumPitch.SD, velocity=100, confidence=0.9),
    DrumEvent(time_sec=1.0, pitch=DrumPitch.BD, velocity=100, confidence=0.9),
    DrumEvent(time_sec=1.5, pitch=DrumPitch.SD, velocity=100, confidence=0.9),
]

grid = TempoGrid(bpm=120.0, time_sig_num=4, time_sig_den=4, beat_times=[], downbeat_times=[0.0, 2.0, 4.0, 6.0], measure_count=3)

measures = quantize(events, grid)

with open('test_output.ly', 'w', encoding='utf-8') as f:
    f.write(_generate_ly_source(measures, "Test", "Test", 120.0))

print("LilyPond Source written to test_output.ly")
