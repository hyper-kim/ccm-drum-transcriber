with open("src/train/train.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

out_lines = []
skip_mode = False
for i, line in enumerate(lines):
    if "# Stage 1: Isolated Hits" in line:
        out_lines.append("    if best_model_path.exists():\n")
        out_lines.append("        logger.info(f'Resuming from checkpoint {best_model_path}. Skipping Stage 1 & 2.')\n")
        out_lines.append("        model.module.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))\n")
        out_lines.append("    else:\n")
        out_lines.append("    " + line)
        skip_mode = True
        continue
        
    if "# Stage 3: Full Context" in line:
        skip_mode = False
        out_lines.append(line)
        continue
        
    if skip_mode:
        out_lines.append("    " + line)
    else:
        out_lines.append(line)

with open("src/train/train.py", "w", encoding="utf-8") as f:
    f.writelines(out_lines)

print("train.py patched to skip Stage 1 and 2 properly!")
