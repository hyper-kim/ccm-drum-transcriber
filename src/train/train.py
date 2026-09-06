import os
import argparse
import logging
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from .dataset import (
    GrooveDrumDataset, IsolatedGrooveDataset, SuperimposedGrooveDataset, 
    download_and_extract, DrumClassMapping
)
from .model import DrumCRNN

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

class UnevenDataParallel(nn.DataParallel):
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
        for i, size in enumerate(dynamic_chunks):
            dev = device_ids[i] if i < len(device_ids) else device_ids[-1]
            chunk = t[start:start+size].to(dev)
            chunks.append(chunk)
            start += size
            
        return tuple((c,) for c in chunks), tuple({} for _ in chunks)

def train_stage(model, dataloaders, criterion, optimizer, scaler, scheduler, device, stage_name, epochs, is_framewise=False, save_path=None):
    logger.info(f"--- Starting {stage_name} ---")
    best_f1 = 0.0
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        
        train_loader = dataloaders['train']
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            with torch.amp.autocast("cuda"):
                logits = model(x)
                
                if not is_framewise:
                    # Pool over time for 200ms isolated/superimposed clips
                    logits = logits.mean(dim=1)
                    loss = criterion(logits, y)
                else:
                    min_len = min(logits.shape[1], y.shape[1])
                    loss = criterion(logits[:, :min_len, :], y[:, :min_len, :])
                    
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            
            if batch_idx % 20 == 0:
                logger.info(f"{stage_name} Epoch [{epoch}/{epochs}] Batch {batch_idx}/{len(train_loader)} Loss: {loss.item():.4f}")
                
        # Eval
        model.eval()
        val_loss = 0.0
        all_preds, all_targets = [], []
        val_loader = dataloaders['val']
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                with torch.amp.autocast("cuda"):
                    logits = model(x)
                    if not is_framewise:
                        logits = logits.mean(dim=1)
                        loss = criterion(logits, y)
                        probs = torch.sigmoid(logits)
                        preds = (probs > 0.5).int().cpu().numpy().reshape(-1)
                        targets = (y > 0.4).int().cpu().numpy().reshape(-1)
                    else:
                        min_len = min(logits.shape[1], y.shape[1])
                        loss = criterion(logits[:, :min_len, :], y[:, :min_len, :])
                        probs = torch.sigmoid(logits[:, :min_len, :])
                        preds = (probs > 0.5).int().cpu().numpy().reshape(-1)
                        targets = (y[:, :min_len, :] > 0.4).int().cpu().numpy().reshape(-1)
                        
                val_loss += loss.item()
                all_preds.extend(preds)
                all_targets.extend(targets)
                
        val_f1 = f1_score(all_targets, all_preds, zero_division=0)
        scheduler.step(val_f1)
        
        logger.info(f"==> {stage_name} Epoch {epoch} Summary: Train Loss: {total_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f} | Val F1: {val_f1:.4f}")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            if save_path:
                model_to_save = model.module if hasattr(model, 'module') else model
                torch.save(model_to_save.state_dict(), save_path)
                logger.info(f"Saved new best model for {stage_name} with F1: {best_f1:.4f}")

def train():
    data_dir = Path("./data")
    download_and_extract(data_dir)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Maximize GPU Memory Utilization (~20GB on 3090, ~14GB on 5060 Ti)
    chunk_sizes = [2457, 1639] # Multi-GPU mapping (3090, 5060 Ti)
    batch_size = 4096
    
    model = DrumCRNN(num_classes=DrumClassMapping.NUM_CLASSES)
    model = UnevenDataParallel(model, chunk_sizes=chunk_sizes).to(device)
    
    save_dir = Path("models")
    save_dir.mkdir(exist_ok=True)
    best_model_path = save_dir / "best_drum_crnn_11class.pt"
    
    pos_weight = torch.ones([DrumClassMapping.NUM_CLASSES]) * 10.0
    pos_weight[0] = 20.0 # 2x Weight for Kick
    pos_weight[4:7] = 5.0 # Half weight for Toms
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    
    if best_model_path.exists():
        logger.info(f'Resuming from checkpoint {best_model_path}. Skipping Stage 1 & 2.')
        model.module.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    else:
        # Stage 1: Isolated Hits
        logger.info("Initializing Stage 1 (Isolated Hits) datasets...")
        train_dl1 = DataLoader(IsolatedGrooveDataset(data_dir, split="train"), batch_size=batch_size, shuffle=True)
        val_dl1 = DataLoader(IsolatedGrooveDataset(data_dir, split="validation"), batch_size=batch_size, shuffle=False)
        train_stage(model, {'train': train_dl1, 'val': val_dl1}, criterion, optimizer, scaler, scheduler, device, "Stage1_Isolated", epochs=15, is_framewise=False, save_path=best_model_path)
        
        # Stage 2: Superimposed Hits
        logger.info("Initializing Stage 2 (Superimposed Hits) datasets...")
        train_dl2 = DataLoader(SuperimposedGrooveDataset(data_dir, split="train"), batch_size=batch_size, shuffle=True)
        val_dl2 = DataLoader(SuperimposedGrooveDataset(data_dir, split="validation"), batch_size=batch_size, shuffle=False)
        train_stage(model, {'train': train_dl2, 'val': val_dl2}, criterion, optimizer, scaler, scheduler, device, "Stage2_Superimposed", epochs=15, is_framewise=False, save_path=best_model_path)
        
    # Stage 3: Full Context
    logger.info("Initializing Stage 3 (Continuous) datasets...")
    batch_size_stage3 = 384 # Perfectly tuned to hit 23GB and 15GB VRAM usage
    train_dl3 = DataLoader(GrooveDrumDataset(data_dir, split="train"), batch_size=batch_size_stage3, shuffle=True)
    val_dl3 = DataLoader(GrooveDrumDataset(data_dir, split="validation"), batch_size=batch_size_stage3, shuffle=False)
    train_stage(model, {'train': train_dl3, 'val': val_dl3}, criterion, optimizer, scaler, scheduler, device, "Stage3_Continuous", epochs=30, is_framewise=True, save_path=best_model_path)

if __name__ == "__main__":
    train()
