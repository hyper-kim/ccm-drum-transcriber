with open("src/score_writer.py", "r", encoding="utf-8") as f:
    text = f.read()

text = text.replace('} \\\\ {', '} \\\\\\\\ {')

with open("src/score_writer.py", "w", encoding="utf-8") as f:
    f.write(text)
print("Fixed backslashes in score_writer.py!")
