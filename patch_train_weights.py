import re

with open("src/train/train.py", "r", encoding="utf-8") as f:
    text = f.read()

old_weight = 'pos_weight = torch.ones([DrumClassMapping.NUM_CLASSES]) * 30.0'
new_weight = '''pos_weight = torch.ones([DrumClassMapping.NUM_CLASSES]) * 30.0
    pos_weight[0] = 60.0 # 2x Weight for Kick to prevent Tom confusion
    pos_weight[4:7] = 15.0 # Half weight for Toms'''

text = text.replace(old_weight, new_weight)

with open("src/train/train.py", "w", encoding="utf-8") as f:
    f.write(text)

print("Class weights updated for Kick emphasis.")
