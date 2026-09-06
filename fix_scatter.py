import re

with open("src/train/train.py", "r", encoding="utf-8") as f:
    text = f.read()

old_udp = '''    def scatter(self, inputs, kwargs, device_ids):
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

new_udp = '''    def scatter(self, inputs, kwargs, device_ids):
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
        for i, size in enumerate(dynamic_chunks):
            dev = device_ids[i] if i < len(device_ids) else device_ids[-1]
            chunk = t[start:start+size].to(dev)
            chunks.append(chunk)
            start += size
            
        return tuple((c,) for c in chunks), tuple({} for _ in chunks)'''

text = text.replace(old_udp, new_udp)

with open("src/train/train.py", "w", encoding="utf-8") as f:
    f.write(text)

print("Fixed device allocation in UnevenDataParallel scatter.")
