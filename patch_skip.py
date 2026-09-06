import re

with open("src/train/train.py", "r", encoding="utf-8") as f:
    text = f.read()

old_s1 = '''    # Stage 1: Isolated Hits
    logger.info("Initializing Stage 1 (Isolated Hits) datasets...")
    train_dl1 = DataLoader(IsolatedGrooveDataset(data_dir, split="train"), batch_size=batch_size, shuffle=True)
    val_dl1 = DataLoader(IsolatedGrooveDataset(data_dir, split="validation"), batch_size=batch_size, shuffle=False)
    
    # Integrate external dataset
    try:
        freesound_dataset = FreesoundDrumDataset(data_dir=None, sr=44100, window_sec=0.10)
        from torch.utils.data import ConcatDataset
        if len(freesound_dataset) > 0:
            train_dl1.dataset = ConcatDataset([train_dl1.dataset, freesound_dataset])
            print("Successfully merged Freesound Dataset into Stage 1!")
    except Exception as e:
        print("Could not load Freesound dataset:", e)
        
    train_stage(model, {'train': train_dl1, 'val': val_dl1}, criterion, optimizer, scaler, scheduler, device, "Stage1_Isolated", epochs=15, is_framewise=False, save_path=best_model_path)
    
    # Stage 2: Superimposed Hits
    logger.info("Initializing Stage 2 (Superimposed Hits) datasets...")
    train_dl2 = DataLoader(SuperimposedGrooveDataset(data_dir, split="train"), batch_size=batch_size, shuffle=True)
    val_dl2 = DataLoader(SuperimposedGrooveDataset(data_dir, split="validation"), batch_size=batch_size, shuffle=False)
    train_stage(model, {'train': train_dl2, 'val': val_dl2}, criterion, optimizer, scaler, scheduler, device, "Stage2_Superimposed", epochs=15, is_framewise=False, save_path=best_model_path)'''

new_s1 = '''    # Load Stage 2 checkpoint if exists
    if best_model_path.exists():
        logger.info(f"Resuming from checkpoint {best_model_path}. Skipping Stage 1 & 2.")
        model.module.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        # Stage 1: Isolated Hits
        logger.info("Initializing Stage 1 (Isolated Hits) datasets...")
        train_dl1 = DataLoader(IsolatedGrooveDataset(data_dir, split="train"), batch_size=batch_size, shuffle=True)
        val_dl1 = DataLoader(IsolatedGrooveDataset(data_dir, split="validation"), batch_size=batch_size, shuffle=False)
        
        # Integrate external dataset
        try:
            freesound_dataset = FreesoundDrumDataset(data_dir=None, sr=44100, window_sec=0.10)
            from torch.utils.data import ConcatDataset
            if len(freesound_dataset) > 0:
                train_dl1.dataset = ConcatDataset([train_dl1.dataset, freesound_dataset])
                print("Successfully merged Freesound Dataset into Stage 1!")
        except Exception as e:
            print("Could not load Freesound dataset:", e)
            
        train_stage(model, {'train': train_dl1, 'val': val_dl1}, criterion, optimizer, scaler, scheduler, device, "Stage1_Isolated", epochs=15, is_framewise=False, save_path=best_model_path)
        
        # Stage 2: Superimposed Hits
        logger.info("Initializing Stage 2 (Superimposed Hits) datasets...")
        train_dl2 = DataLoader(SuperimposedGrooveDataset(data_dir, split="train"), batch_size=batch_size, shuffle=True)
        val_dl2 = DataLoader(SuperimposedGrooveDataset(data_dir, split="validation"), batch_size=batch_size, shuffle=False)
        train_stage(model, {'train': train_dl2, 'val': val_dl2}, criterion, optimizer, scaler, scheduler, device, "Stage2_Superimposed", epochs=15, is_framewise=False, save_path=best_model_path)'''

text = text.replace(old_s1, new_s1)

with open("src/train/train.py", "w", encoding="utf-8") as f:
    f.write(text)

print("train.py patched to skip Stage 1 & 2 if checkpoint exists.")
