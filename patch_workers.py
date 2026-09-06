import re

with open("src/train/train.py", "r", encoding="utf-8") as f:
    text = f.read()

# Replace DataLoader(...) with DataLoader(..., num_workers=8, pin_memory=True, persistent_workers=True)
text = re.sub(
    r'(DataLoader\([^,]+, batch_size=[^,]+, shuffle=(?:True|False))(?!, num_workers)',
    r'\1, num_workers=8, pin_memory=True, persistent_workers=True',
    text
)

with open("src/train/train.py", "w", encoding="utf-8") as f:
    f.write(text)

print("DataLoader multiprocessing parameters added!")
