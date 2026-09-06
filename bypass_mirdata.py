import json
import os

path = r'D:\tmp\mir_datasets\freesound_one_shot_percussive_sounds\sound_info_analysis.json'
if os.path.exists(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print("Found JSON, number of keys:", len(data))
    first_key = list(data.keys())[0]
    print(f"Sample info for {first_key}:", data[first_key]['tags'])
else:
    print("JSON not found at", path)
