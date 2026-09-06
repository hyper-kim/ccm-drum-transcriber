import re

with open("src/train/dataset.py", "r", encoding="utf-8") as f:
    text = f.read()

old_func = '''def parse_midi_onsets(midi_path: str, duration_sec: float, fps: int = 100) -> np.ndarray:
    mid = mido.MidiFile(midi_path)
    num_frames = int(np.ceil(duration_sec * fps))
    roll = np.zeros((num_frames, DrumClassMapping.NUM_CLASSES), dtype=np.float32)
    
    current_time = 0.0
    for track in mid.tracks:
        for msg in track:
            current_time += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                if getattr(msg, 'channel', 9) == 9: # Drum Channel
                    if msg.note in DrumClassMapping.MIDI_MAP:
                        cls_idx = DrumClassMapping.MIDI_MAP[msg.note]
                        frame_idx = int(round(current_time * fps))
                        if frame_idx < num_frames:
                            roll[frame_idx, cls_idx] = 1.0
    return roll'''

new_func = '''def parse_midi_onsets(midi_path: str, duration_sec: float, fps: int = 100) -> np.ndarray:
    mid = mido.MidiFile(midi_path)
    num_frames = int(np.ceil(duration_sec * fps))
    roll = np.zeros((num_frames, DrumClassMapping.NUM_CLASSES), dtype=np.float32)
    
    # Gaussian kernel for temporal label smoothing
    kernel = np.array([0.1, 0.5, 1.0, 0.5, 0.1], dtype=np.float32)
    kernel_radius = len(kernel) // 2
    
    current_time = 0.0
    for track in mid.tracks:
        for msg in track:
            current_time += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                if getattr(msg, 'channel', 9) == 9: # Drum Channel
                    if msg.note in DrumClassMapping.MIDI_MAP:
                        cls_idx = DrumClassMapping.MIDI_MAP[msg.note]
                        frame_idx = int(round(current_time * fps))
                        
                        # Apply label smoothing
                        for i, val in enumerate(kernel):
                            target_frame = frame_idx - kernel_radius + i
                            if 0 <= target_frame < num_frames:
                                roll[target_frame, cls_idx] = max(roll[target_frame, cls_idx], val)
    return roll'''

text = text.replace(old_func, new_func)

with open("src/train/dataset.py", "w", encoding="utf-8") as f:
    f.write(text)

print("Updated parse_midi_onsets in dataset.py with label smoothing")
