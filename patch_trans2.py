with open('src/transcriber.py', 'r', encoding='utf-8') as f:
    text = f.read()

target = '''    max_val = torch.max(torch.abs(y))
    if max_val > 0:
        y = y / max_val'''
replacement = ""
text = text.replace(target, replacement)

target2 = '''onset_threshold: float = 0.1,'''
replacement2 = '''onset_threshold: float = 0.02,'''
text = text.replace(target2, replacement2)

with open('src/transcriber.py', 'w', encoding='utf-8') as f:
    f.write(text)
print("Removed peak norm, lowered threshold to 0.02")
