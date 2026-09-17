"""
src/train/train.py  (v7 — 멀티 GPU: 연산 속도 비례 주입)

이전 병목 원인: VRAM 비율(3:2)로 배치를 나누면, 실제 연산 능력이 월등히 뛰어난
3090(GPU0)이 자기 몫을 끝내고 5060Ti(GPU1)를 기다리며 놀게 됩니다.
해결: 연산 코어 성능(Compute ratio)에 비례하여 배치 사이즈를 할당합니다.
RTX 3090 vs RTX 5060 Ti의 실질 연산 속도는 약 4:1 ~ 3:1 수준이므로,
주입 속도(chunk sizes)를 [2048, 512]로 설정하여 GPU0을 100% 가동시킵니다.
"""

import hashlib
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader

from .dataset import DrumClassMapping, build_dataset
from .model import DrumHitClassifier

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# UnevenDataParallel — 연산 속도(Compute Speed) 비례 배치 주입
# ---------------------------------------------------------------------------

class UnevenDataParallel(nn.DataParallel):
    def __init__(self, module, chunk_sizes: list[int], **kwargs):
        super().__init__(module, **kwargs)
        self.chunk_sizes = chunk_sizes

    def scatter(self, inputs, kwargs, device_ids):
        x = inputs[0]
        b = x.size(0)
        total = sum(self.chunk_sizes)
        # 사전에 정의된 chunk_sizes 비율대로 배치 분할
        sizes = [max(1, round(b * s / total)) for s in self.chunk_sizes]
        sizes[-1] = b - sum(sizes[:-1])
        if any(s <= 0 for s in sizes):
            return super().scatter(inputs, kwargs, device_ids)
        chunks, start = [], 0
        for i, sz in enumerate(sizes):
            chunks.append(x[start:start + sz].to(device_ids[i]))
            start += sz
        return tuple((c,) for c in chunks), tuple({} for _ in chunks)


def log_vram():
    for i in range(torch.cuda.device_count()):
        alloc  = torch.cuda.memory_allocated(i)  // 1024**2
        reserv = torch.cuda.memory_reserved(i)   // 1024**2
        total  = torch.cuda.get_device_properties(i).total_memory // 1024**2
        name   = torch.cuda.get_device_properties(i).name
        logger.info(f"  VRAM GPU{i} [{name}]: {alloc}MB alloc / {reserv}MB reserved / {total}MB total")


def train():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count()
    
    for i in range(n_gpus):
        p = torch.cuda.get_device_properties(i)
        logger.info(f"Device {i}: {p.name} ({p.total_memory // 1024**3} GB)")

    # ---- Datasets ----
    logger.info("Loading datasets ...")
    data_dir = Path("./data")
    train_ds = build_dataset(data_dir, split="train")
    val_ds   = build_dataset(data_dir, split="validation")

    # 연산 성능(TFLOPS/Cores) 비례 주입 (3090은 5060Ti 대비 약 3~4배 빠름)
    # GPU0: 2048, GPU1: 640 -> Total: 2688
    if n_gpus >= 2:
        batch_size  = 2688
        chunk_sizes = [2048, 640]  # GPU0을 극대화하고 GPU1이 보조하도록 주입 속도 차등
        num_workers = 16
    elif n_gpus == 1:
        batch_size  = 2048
        chunk_sizes = None
        num_workers = 12
    else:
        batch_size  = 128
        chunk_sizes = None
        num_workers = 4

    logger.info(f"Batch size: {batch_size} | Workers: {num_workers} | GPUs: {n_gpus}")

    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(n_gpus > 0),
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )
    val_dl = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(n_gpus > 0),
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )

    # ---- Model ----
    model = DrumHitClassifier(num_classes=DrumClassMapping.NUM_CLASSES)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model params: {total_params:,}  ({total_params/1e6:.1f}M)")

    if n_gpus >= 2 and chunk_sizes:
        logger.info(f"UnevenDataParallel: chunk_sizes={chunk_sizes} (sum={sum(chunk_sizes)}) -> Compute-based load balancing")
        model = UnevenDataParallel(model, chunk_sizes=chunk_sizes, device_ids=list(range(n_gpus)))
    
    model = model.to(device)
    log_vram()

    # ---- Loss ----
    class_weights = torch.tensor(
        #  Kick  Snare  HH-C  HH-O  TomH  TomM  TomL  Crash  Ride  R-Bell  Rimshot
        [  1.0,  1.0,   1.0,  4.0,  2.0,  2.5,  2.0,  4.0,   1.5,  5.0,    3.0  ],
        dtype=torch.float32, device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    epochs    = 100
    scaler    = torch.amp.GradScaler("cuda", enabled=(n_gpus > 0))
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=3e-4,
        epochs=epochs, steps_per_epoch=len(train_dl),
        pct_start=0.1, anneal_strategy="cos",
    )

    save_dir  = Path("models")
    save_dir.mkdir(exist_ok=True)
    best_path = save_dir / "best_drum_classifier.pt"
    best_f1   = 0.0

    logger.info(f"Training {DrumClassMapping.NUM_CLASSES} classes for {epochs} epochs")
    logger.info(f"Classes: {DrumClassMapping.CLASS_NAMES}")
    logger.info(f"Train: {len(train_ds):,} | Val: {len(val_ds):,}")

    for epoch in range(1, epochs + 1):
        # ---- Train ----
        model.train()
        total_loss = 0.0
        n_batches  = len(train_dl)

        for i, (x, y) in enumerate(train_dl):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(n_gpus > 0)):
                logits = model(x)
                loss   = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item()
            if i % 10 == 0:
                logger.info(f"Ep[{epoch}/{epochs}] {i}/{n_batches} loss={loss.item():.4f}")

        avg_loss = total_loss / n_batches

        # ---- Validation ----
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for x, y in val_dl:
                x = x.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=(n_gpus > 0)):
                    logits = model(x)
                all_preds.extend(logits.argmax(dim=1).cpu().tolist())
                all_targets.extend(y.tolist())

        macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
        logger.info(f"Ep[{epoch}/{epochs}] avg_loss={avg_loss:.4f} | macro_F1={macro_f1:.4f}")

        # VRAM 로깅 (5 epoch마다)
        if epoch % 5 == 0:
            log_vram()

        # per-class report (10 epoch마다)
        if epoch % 10 == 0:
            report = classification_report(
                all_targets, all_preds,
                target_names=DrumClassMapping.CLASS_NAMES,
                zero_division=0,
            )
            logger.info(f"\n{report}")

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            m_save = model.module if hasattr(model, "module") else model
            torch.save(m_save.state_dict(), best_path)
            logger.info(f"  -> Best saved (F1={best_f1:.4f}): {best_path}")

    # ---- 완료 ----
    if best_path.exists():
        sha = hashlib.sha256(best_path.read_bytes()).hexdigest()
        logger.info("=" * 60)
        logger.info(f"Done! Best macro-F1: {best_f1:.4f}")
        logger.info(f"SHA256: {sha}")
        logger.info("Run:  git add models/best_drum_classifier.pt && git push")
        logger.info("=" * 60)


if __name__ == "__main__":
    train()
