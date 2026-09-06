import os
import logging
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import ASTForAudioClassification, ASTFeatureExtractor
from sklearn.metrics import f1_score

from .dataset import ASTGrooveDrumDataset, download_and_extract, DrumClassMapping

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

class UnevenDataParallel(nn.DataParallel):
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
            actual_size = min(size, t.size(0) - start)
            chunks.append(t[start:start+actual_size].to(device_ids[i]))
            start += actual_size
        return tuple((c,) for c in chunks), tuple({} for _ in chunks)

def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['input_values'].to(device)
            labels = batch['labels'].to(device)
            
            with torch.amp.autocast("cuda"):
                outputs = model(inputs)
                logits = outputs.logits
            
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).int().cpu().numpy()
            targets = labels.int().cpu().numpy()
            
            all_preds.extend(preds)
            all_targets.extend(targets)
            
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    return f1

def train():
    data_dir = Path("./data")
    download_and_extract(data_dir)
    
    batch_size = 80
    epochs = 30
    accumulation_steps = 4
    lr = 5e-5
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_gpus = torch.cuda.device_count()
    logger.info(f"Using device: {device} with {num_gpus} GPUs (Uneven Batch Split)")
    
    model_name = "MIT/ast-finetuned-audioset-10-10-0.4593"
    feature_extractor = ASTFeatureExtractor.from_pretrained(model_name)
    
    model = ASTForAudioClassification.from_pretrained(
        model_name, 
        num_labels=DrumClassMapping.NUM_CLASSES,
        problem_type="multi_label_classification",
        ignore_mismatched_sizes=True
    )
    
    if num_gpus > 1:
        # Assuming GPU0 is 3090 and GPU1 is 5060 Ti
        chunk_sizes = [48, 32]
        model = UnevenDataParallel(model, chunk_sizes=chunk_sizes)
    model = model.to(device)
    
    logger.info("Initializing datasets...")
    train_dataset = ASTGrooveDrumDataset(data_dir, split="train", feature_extractor=feature_extractor)
    val_dataset = ASTGrooveDrumDataset(data_dir, split="validation", feature_extractor=feature_extractor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")
    
    pos_weight = torch.ones([DrumClassMapping.NUM_CLASSES]).to(device) * 10.0
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    save_dir = Path("models")
    save_dir.mkdir(exist_ok=True)
    best_model_path = save_dir / "best_ast_drum"
    best_f1 = 0.0
    
    logger.info("Starting AST training loop (UnevenDataParallel)...")
    optimizer.zero_grad()
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            inputs = batch['input_values'].to(device)
            labels = batch['labels'].to(device)
            
            with torch.amp.autocast("cuda"):
                outputs = model(inputs)
                loss = criterion(outputs.logits, labels) / accumulation_steps
                
            scaler.scale(loss).backward()
            
            total_loss += loss.item()
            
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            if batch_idx % 5 == 0:
                logger.info(f"Epoch [{epoch}/{epochs}] Batch {batch_idx}/{len(train_loader)} Loss: {loss.item() * accumulation_steps:.4f}")
                
        avg_train_loss = total_loss / len(train_loader) * accumulation_steps
        val_f1 = evaluate(model, val_loader, device)
        
        logger.info(f"==> Epoch {epoch} Summary: Train Loss: {avg_train_loss:.4f} | Val Macro F1: {val_f1:.4f}")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(best_model_path)
            feature_extractor.save_pretrained(best_model_path)
            logger.info(f"Saved new best AST model with F1: {best_f1:.4f}")

if __name__ == "__main__":
    train()
