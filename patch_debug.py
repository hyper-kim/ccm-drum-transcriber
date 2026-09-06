import re

with open("src/transcriber.py", "r", encoding="utf-8") as f:
    text = f.read()

old_loop = '''    for cls_idx in range(DrumClassMapping.NUM_CLASSES):
        cls_probs = probs[:, cls_idx]
        peaks, _ = scipy.signal.find_peaks(cls_probs, height=onset_threshold, distance=min_dist_frames)
        
        rep_note = reverse_map[cls_idx][0]
        try:
            pitch = DrumPitch(rep_note)
        except ValueError:
            continue
            
        for p in peaks:'''

new_loop = '''    for cls_idx in range(DrumClassMapping.NUM_CLASSES):
        cls_probs = probs[:, cls_idx]
        peaks, _ = scipy.signal.find_peaks(cls_probs, height=onset_threshold, distance=min_dist_frames)
        
        print(f"Class {cls_idx} max prob: {cls_probs.max():.4f}, peaks found: {len(peaks)}")
        
        rep_note = reverse_map[cls_idx][0]
        try:
            pitch = DrumPitch(rep_note)
        except ValueError:
            print(f"Class {cls_idx} ValueError for rep_note {rep_note}!")
            continue
            
        for p in peaks:'''

text = text.replace(old_loop, new_loop)

with open("src/transcriber.py", "w", encoding="utf-8") as f:
    f.write(text)

print("Added debug prints to transcriber.py")
