import pandas as pd
from pathlib import Path
import mido

data_dir = Path("data/groove")
df = pd.read_csv(data_dir / "info.csv")

isolated_counts = {}

for _, row in df.iterrows():
    midi_path = data_dir / row['midi_filename']
    if not midi_path.exists(): continue
    try:
        mid = mido.MidiFile(midi_path)
        notes = []
        current_time = 0.0
        for track in mid.tracks:
            for msg in track:
                current_time += msg.time
                if msg.type == 'note_on' and msg.velocity > 0:
                    notes.append({'pitch': msg.note, 'time': current_time})
        
        notes.sort(key=lambda n: n['time'])
        for i in range(len(notes)):
            n = notes[i]
            is_isolated = True
            # Check previous note
            if i > 0 and n['time'] - notes[i-1]['time'] < 0.1: # less than 100ms
                is_isolated = False
            # Check next note
            if i < len(notes)-1 and notes[i+1]['time'] - n['time'] < 0.1:
                is_isolated = False
            
            if is_isolated:
                isolated_counts[n['pitch']] = isolated_counts.get(n['pitch'], 0) + 1
    except Exception as e:
        pass

print("Isolated notes count by pitch:", sorted(isolated_counts.items()))
