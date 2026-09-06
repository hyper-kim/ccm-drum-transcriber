import re

with open("src/train/train.py", "r", encoding="utf-8") as f:
    text = f.read()

old_metrics_1 = '''                        preds = (probs > 0.5).int().cpu().numpy().reshape(-1)
                        targets = y.int().cpu().numpy().reshape(-1)'''

new_metrics_1 = '''                        preds = (probs > 0.5).int().cpu().numpy().reshape(-1)
                        targets = (y > 0.4).int().cpu().numpy().reshape(-1)'''

text = text.replace(old_metrics_1, new_metrics_1)

old_metrics_2 = '''                        preds = (probs > 0.5).int().cpu().numpy().reshape(-1)
                        targets = y[:, :min_len, :].int().cpu().numpy().reshape(-1)'''

new_metrics_2 = '''                        preds = (probs > 0.5).int().cpu().numpy().reshape(-1)
                        targets = (y[:, :min_len, :] > 0.4).int().cpu().numpy().reshape(-1)'''

text = text.replace(old_metrics_2, new_metrics_2)

old_f1 = '''                val_f1 = f1_score(all_targets, all_preds, zero_division=0)'''
new_f1 = '''                val_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)'''

text = text.replace(old_f1, new_f1)

with open("src/train/train.py", "w", encoding="utf-8") as f:
    f.write(text)

print("Updated train.py metrics calculation")
