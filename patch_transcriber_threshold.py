import re

with open("src/transcriber.py", "r", encoding="utf-8") as f:
    text = f.read()

text = text.replace("onset_threshold: float = 0.5", "onset_threshold: float = 0.1")

with open("src/transcriber.py", "w", encoding="utf-8") as f:
    f.write(text)

print("Updated default onset_threshold in transcriber.py to 0.1")
