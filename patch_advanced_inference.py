import re

with open("src/transcriber.py", "r", encoding="utf-8") as f:
    text = f.read()

old_infer = '''    y_dev = y.to(device)
    mel_spec = amp_to_db(mel_transform(y_dev)).unsqueeze(0) # [1, 1, 229, T]
    
    segment_frames = 300 # 3
    step_frames = 150 # 1.5 ħ
    total_frames = mel_spec.shape[3]
    
    probs = np.zeros((total_frames, DrumClassMapping.NUM_CLASSES), dtype=np.float32)
    counts = np.zeros((total_frames, 1), dtype=np.float32)
    
    with torch.no_grad():
        for start_f in range(0, total_frames, step_frames):
            end_f = min(start_f + segment_frames, total_frames)
            seg_len = end_f - start_f
            
            x = mel_spec[:, :, :, start_f:end_f]
            if x.shape[3] < segment_frames:
                pad = segment_frames - x.shape[3]
                x = torch.nn.functional.pad(x, (0, pad))
                
            logits = model(x)
            seg_probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
            
            probs[start_f:end_f] += seg_probs[:seg_len]
            counts[start_f:end_f] += 1.0
            
    probs /= np.maximum(counts, 1.0)'''

new_infer = '''    y_dev = y.to(device)
    mel_spec_orig = amp_to_db(mel_transform(y_dev)).unsqueeze(0) # [1, 1, 229, T]
    
    # 1. Cymbal Unmasking (High-Pass Filter Ensemble)
    import scipy.signal
    y_np = y.cpu().numpy()[0]
    b, a = scipy.signal.butter(4, 4000, btype='high', fs=sr)
    y_hpf_np = scipy.signal.lfilter(b, a, y_np)
    y_hpf_dev = torch.from_numpy(y_hpf_np.astype("float32")).unsqueeze(0).to(device)
    mel_spec_hpf = amp_to_db(mel_transform(y_hpf_dev)).unsqueeze(0)
    
    segment_frames = 300
    step_frames = 150
    total_frames = mel_spec_orig.shape[3]
    
    probs_orig = np.zeros((total_frames, DrumClassMapping.NUM_CLASSES), dtype=np.float32)
    probs_hpf = np.zeros((total_frames, DrumClassMapping.NUM_CLASSES), dtype=np.float32)
    counts = np.zeros((total_frames, 1), dtype=np.float32)
    
    with torch.no_grad():
        for start_f in range(0, total_frames, step_frames):
            end_f = min(start_f + segment_frames, total_frames)
            seg_len = end_f - start_f
            
            x_orig = mel_spec_orig[:, :, :, start_f:end_f]
            x_hpf = mel_spec_hpf[:, :, :, start_f:end_f]
            
            if x_orig.shape[3] < segment_frames:
                pad = segment_frames - x_orig.shape[3]
                x_orig = torch.nn.functional.pad(x_orig, (0, pad))
                x_hpf = torch.nn.functional.pad(x_hpf, (0, pad))
                
            probs_orig[start_f:end_f] += torch.sigmoid(model(x_orig)).squeeze(0).cpu().numpy()[:seg_len]
            probs_hpf[start_f:end_f] += torch.sigmoid(model(x_hpf)).squeeze(0).cpu().numpy()[:seg_len]
            counts[start_f:end_f] += 1.0
            
    probs_orig /= np.maximum(counts, 1.0)
    probs_hpf /= np.maximum(counts, 1.0)
    
    # Merge probabilities: For cymbals, use max to prevent masking by loud drums
    probs = np.copy(probs_orig)
    cymbal_classes = [3, 7, 8, 9, 10] # OpenHH, Crash1, Crash2, Splash, Ride
    for c_idx in cymbal_classes:
        probs[:, c_idx] = np.maximum(probs_orig[:, c_idx], probs_hpf[:, c_idx])'''

old_post = '''    events.sort(key=lambda e: e.time_sec)
    
    logger.info(f" ߷  Ϸ:  {len(events)} Ʈ ")'''

new_post = '''    events.sort(key=lambda e: e.time_sec)
    
    # 2. Kick vs Large Tom Context HMM (Post-processing)
    tom_times = [e.time_sec for e in events if e.pitch in {DrumPitch.TOM_H, DrumPitch.TOM_M, DrumPitch.TOM_F}]
    for ev in events:
        if ev.pitch == DrumPitch.TOM_F:
            # Check if this Large Tom is isolated (no other toms within +/- 2.0 seconds)
            nearby_toms = [t for t in tom_times if t != ev.time_sec and abs(t - ev.time_sec) < 2.0]
            if len(nearby_toms) == 0:
                logger.debug(f"Context Filter: Converted isolated Large Tom at {ev.time_sec:.2f}s to Kick.")
                ev.pitch = DrumPitch.BD
                
    logger.info(f"Cymbal Unmasking & Context Filtering Applied. Total events: {len(events)}")'''

text = text.replace(old_infer, new_infer)
text = text.replace(old_post, new_post)

with open("src/transcriber.py", "w", encoding="utf-8") as f:
    f.write(text)

print("transcriber.py patched for advanced inference.")
