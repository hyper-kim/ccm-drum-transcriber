import re

with open("src/train/train.py", "r", encoding="utf-8") as f:
    text = f.read()

old_batch = '''    chunk_sizes = [96, 64] # Multi-GPU mapping (3090, 5060 Ti)
    batch_size = 160'''
new_batch = '''    # Maximize GPU Memory Utilization (~20GB on 3090, ~14GB on 5060 Ti)
    chunk_sizes = [2457, 1639] # Multi-GPU mapping (3090, 5060 Ti)
    batch_size = 4096'''

text = text.replace(old_batch, new_batch)

with open("src/train/train.py", "w", encoding="utf-8") as f:
    f.write(text)

print("Batch size updated.")
