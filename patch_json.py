import re

with open("src/train/dataset.py", "r", encoding="utf-8") as f:
    text = f.read()

old_json = '''            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)'''
                
new_json = '''            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                print(f"[FreesoundDataset] Failed to load JSON: {e}")
                data = {}'''

text = text.replace(old_json, new_json)

with open("src/train/dataset.py", "w", encoding="utf-8") as f:
    f.write(text)

print("Added try/except block for JSON loading.")
