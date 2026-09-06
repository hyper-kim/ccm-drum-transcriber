import re

with open("src/transcriber.py", "r", encoding="utf-8") as f:
    text = f.read()

old_logic = '''            probs[start_f:end_f] += seg_probs[:seg_len]
            counts[start_f:end_f] += 1.0
            
    probs /= np.maximum(counts, 1.0)'''

new_logic = '''            probs[start_f:end_f] = np.maximum(probs[start_f:end_f], seg_probs[:seg_len])
            
    # No averaging needed, we just take the max probability across overlapping segments'''

text = text.replace(old_logic, new_logic)

with open("src/transcriber.py", "w", encoding="utf-8") as f:
    f.write(text)

print("Updated transcriber.py to use np.maximum instead of averaging.")
