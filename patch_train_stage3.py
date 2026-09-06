import re

with open("src/train/train.py", "r", encoding="utf-8") as f:
    text = f.read()

# Fix UnevenDataParallel to be proportional
old_udp = '''class UnevenDataParallel(nn.DataParallel):
    def __init__(self, module, chunk_sizes, *args, **kwargs):
        super().__init__(module, *args, **kwargs)
        self.chunk_sizes = chunk_sizes

    def scatter(self, inputs, kwargs, device_ids):
        t = inputs[0]
        chunks = []
        start = 0
        for i, size in enumerate(self.chunk_sizes):
            if start >= t.size(0):
                break
            end = min(start + size, t.size(0))
            chunks.append(t[start:end])
            start = end
            
        return tuple((c,) for c in chunks), tuple({} for _ in chunks)'''

new_udp = '''class UnevenDataParallel(nn.DataParallel):
    def __init__(self, module, chunk_sizes, *args, **kwargs):
        super().__init__(module, *args, **kwargs)
        self.base_chunk_sizes = chunk_sizes

    def scatter(self, inputs, kwargs, device_ids):
        t = inputs[0]
        b = t.size(0)
        
        # Proportional chunks based on actual batch size
        total_base = sum(self.base_chunk_sizes)
        dynamic_chunks = [int((s / total_base) * b) for s in self.base_chunk_sizes]
        dynamic_chunks[-1] = b - sum(dynamic_chunks[:-1]) # Fix rounding error
        
        # Fallback to standard if too small to split
        if any(c <= 0 for c in dynamic_chunks):
            return super().scatter(inputs, kwargs, device_ids)
            
        chunks = []
        start = 0
        for size in dynamic_chunks:
            chunks.append(t[start:start+size])
            start += size
            
        return tuple((c,) for c in chunks), tuple({} for _ in chunks)'''

if "def scatter(self, inputs, kwargs, device_ids):" in text:
    text = re.sub(r'class UnevenDataParallel\(nn\.DataParallel\):.*?return tuple\(\(c,\) for c in chunks\), tuple\(\{\} for _ in chunks\)', new_udp, text, flags=re.DOTALL)


old_stage3 = '''    # Stage 3: Full Context
    logger.info("Initializing Stage 3 (Continuous) datasets...")
    train_dl3 = DataLoader(GrooveDrumDataset(data_dir, split="train"), batch_size=batch_size, shuffle=True)
    val_dl3 = DataLoader(GrooveDrumDataset(data_dir, split="validation"), batch_size=batch_size, shuffle=False)
    train_stage(model, {'train': train_dl3, 'val': val_dl3}, criterion, optimizer, scaler, scheduler, device, "Stage3_Continuous", epochs=30, is_framewise=True, save_path=best_model_path)'''

new_stage3 = '''    # Stage 3: Full Context
    logger.info("Initializing Stage 3 (Continuous) datasets...")
    batch_size_stage3 = 64 # Prevent OOM due to large continuous tensors
    train_dl3 = DataLoader(GrooveDrumDataset(data_dir, split="train"), batch_size=batch_size_stage3, shuffle=True)
    val_dl3 = DataLoader(GrooveDrumDataset(data_dir, split="validation"), batch_size=batch_size_stage3, shuffle=False)
    train_stage(model, {'train': train_dl3, 'val': val_dl3}, criterion, optimizer, scaler, scheduler, device, "Stage3_Continuous", epochs=30, is_framewise=True, save_path=best_model_path)'''

text = text.replace(old_stage3, new_stage3)

with open("src/train/train.py", "w", encoding="utf-8") as f:
    f.write(text)

print("train.py patched for Stage 3 batch size and proportional UDP.")
