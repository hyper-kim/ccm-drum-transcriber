import re

with open("src/train/train.py", "r", encoding="utf-8") as f:
    text = f.read()

# I want to lower pos_weight from 30.0 to 10.0 because label smoothing makes the targets much denser!
old_pos = '''    pos_weight = torch.ones([DrumClassMapping.NUM_CLASSES]) * 30.0
    pos_weight[0] = 60.0 # 2x Weight for Kick to prevent Tom confusion
    pos_weight[4:7] = 15.0 # Half weight for Toms'''

new_pos = '''    pos_weight = torch.ones([DrumClassMapping.NUM_CLASSES]) * 10.0
    pos_weight[0] = 20.0 # 2x Weight for Kick
    pos_weight[4:7] = 5.0 # Half weight for Toms'''

text = text.replace(old_pos, new_pos)

# Also enforce running ONLY Stage 3 if the file exists
old_logic = '''if Path(best_model_path).exists():
        logger.info(f"Resuming from existing model: {best_model_path}")
        model.module.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    else:'''

new_logic = '''if Path(best_model_path).exists():
        logger.info(f"Resuming from existing model: {best_model_path}")
        model.module.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    
    # We always run Stage 3, skip 1 and 2 if exists
    if not Path(best_model_path).exists():'''

text = text.replace(old_logic, new_logic)

with open("src/train/train.py", "w", encoding="utf-8") as f:
    f.write(text)

print("Updated pos_weight and forced Stage 3 execution")
