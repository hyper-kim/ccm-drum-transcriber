import re

with open("src/train/train.py", "r", encoding="utf-8") as f:
    text = f.read()

old_bs = "batch_size_stage3 = 192 # Sweet spot for GRU sequence length 300 on 24GB + 16GB VRAM"
new_bs = "batch_size_stage3 = 320 # Pushed to the absolute maximum to fully saturate 24GB+16GB VRAM"

text = text.replace(old_bs, new_bs)

with open("src/train/train.py", "w", encoding="utf-8") as f:
    f.write(text)

print("Batch size adjusted to 320")
