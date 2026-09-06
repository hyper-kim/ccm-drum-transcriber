import re

with open("main.py", "r", encoding="utf-8") as f:
    text = f.read()

old_call = "events = transcribe_drums(drums_wav, None, onset_threshold=0.2, min_onset_gap_sec=0.05, verbose=args.verbose)"
new_call = "events = transcribe_drums(drums_wav, None, onset_threshold=0.05, min_onset_gap_sec=0.05, verbose=args.verbose)"

text = text.replace(old_call, new_call)

with open("main.py", "w", encoding="utf-8") as f:
    f.write(text)

print("Updated onset_threshold in main.py to 0.05")
