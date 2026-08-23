"""
training.py
===========
Train the denoising network on patches produced by prepare_dataset.py.

    python training.py                          # LightUNet, default settings
    python training.py --model encoderdecoder   # ablation without skips
    python training.py --epochs 300 --lr 1e-4

The network is trained on zero-mean residuals: the local reference mean was
removed from both input and target during extraction, so absolute depth is
never presented to it and cannot be altered by it.

Both architectures must stay identical apart from the skip connections, or
the ablation stops being a controlled comparison.
"""

import os
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from .config import Config
from .dataset import load_split
from .model import LightUNet, EncoderDecoder

MODELS = {"lightunet": LightUNet, "encoderdecoder": EncoderDecoder}


# ─────────────────────────────────────────────────────────────────────────────
def pick_device(prefer=None):
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────────────────
class PatchDataset(Dataset):
    """Patch pairs with optional dihedral augmentation.

    The same transform is applied to input and target, since the two are
    spatially aligned and the task is dense regression rather than
    classification.

    Rotations are included as well as flips. Survey lines run in different
    directions from one survey to the next, so the noise patterns the network
    has to suppress are not tied to a fixed orientation, and teaching it to
    handle any orientation is closer to how it will be used than restricting
    it to the track direction of one survey.

    Augmentation is applied per access rather than materialised in advance,
    so the patch counts reported by prepare_dataset.py are the true numbers
    of distinct seabed windows.
    """

    def __init__(self, X, Y, augment=False):
        self.X = torch.from_numpy(np.ascontiguousarray(X)).float()
        self.Y = torch.from_numpy(np.ascontiguousarray(Y)).float()
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x, y = self.X[i], self.Y[i]
        if self.augment:
            k = int(torch.randint(0, 4, (1,)))
            if k:
                x = torch.rot90(x, k, dims=(-2, -1))
                y = torch.rot90(y, k, dims=(-2, -1))
            if torch.rand(1) < 0.5:
                x = torch.flip(x, dims=(-1,))
                y = torch.flip(y, dims=(-1,))
        return x.unsqueeze(0), y.unsqueeze(0)


# ─────────────────────────────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, device, optimiser=None):
    """One pass over a loader. Trains when an optimiser is given."""
    training = optimiser is not None
    model.train(training)
    total, n = 0.0, 0

    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            if training:
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                optimiser.step()
            total += loss.item() * x.size(0)
            n += x.size(0)

    return total / max(n, 1)


def train(cfg, device, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    set_seed(cfg.seed)

    data, pinfo = load_split(cfg.patch_dir)
    Xtr, Ytr, _, _ = data["train"]
    Xva, Yva, _, _ = data["val"]
    print(f"Patches: {len(Xtr)} train, {len(Xva)} val (test held out until evaluation)")
    if len(Xva) < 40:
        print(
            "  NOTE: the validation set is small, so the validation curve "
            "will be noisy and early stopping may fire on noise."
        )

    train_dl = DataLoader(
        PatchDataset(Xtr, Ytr, cfg.augment),
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_dl = DataLoader(
        PatchDataset(Xva, Yva, augment=False), batch_size=cfg.batch_size, shuffle=False
    )

    model = MODELS[cfg.model]().to(device)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {cfg.model}, {n_par:,} trainable parameters, on {device}")

    criterion = nn.L1Loss()
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=cfg.lr_factor, patience=cfg.lr_patience
    )

    ckpt = os.path.join(out_dir, f"best_{cfg.model}.pth")
    best, best_epoch, stale, history = np.inf, 0, 0, []
    t0 = time.time()

    for epoch in range(1, cfg.epochs + 1):
        tr = run_epoch(model, train_dl, criterion, device, optimiser)
        va = run_epoch(model, val_dl, criterion, device)
        scheduler.step(va)
        lr_now = optimiser.param_groups[0]["lr"]
        history.append({"epoch": epoch, "train_loss": tr, "val_loss": va, "lr": lr_now})

        improved = va < best - cfg.min_delta
        if improved:
            best, best_epoch, stale = va, epoch, 0
            torch.save(model.state_dict(), ckpt)
        else:
            stale += 1

        if epoch % 5 == 0 or improved or epoch == 1:
            print(
                f"  epoch {epoch:3d}  train {tr * 100:6.3f} cm  "
                f"val {va * 100:6.3f} cm  lr {lr_now:.2e}"
                f"{'  *' if improved else ''}"
            )

        if stale >= cfg.patience:
            print(
                f"\nEarly stopping at epoch {epoch}: no improvement for "
                f"{cfg.patience} epochs"
            )
            break

    mins = (time.time() - t0) / 60
    print(f"\nBest validation L1 {best * 100:.3f} cm at epoch {best_epoch}")
    print(f"Trained for {mins:.1f} min, weights at {ckpt}")

    # A widening gap between the curves is the signal to watch given the
    # small training set; record both so it can be inspected afterwards.
    final = history[-1]
    gap = (final["val_loss"] - final["train_loss"]) / max(final["train_loss"], 1e-9)
    if gap > 0.5:
        print(
            f"  NOTE: final validation loss exceeds training loss by "
            f"{gap * 100:.0f}%, which suggests overfitting. Consider more "
            f"regularisation or a smaller model."
        )

    summary = {
        "model": cfg.model,
        "n_parameters": int(n_par),
        "best_val_loss": float(best),
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "minutes": round(mins, 2),
        "device": str(device),
        "n_train": int(len(Xtr)),
        "n_val": int(len(Xva)),
        "patch_info": pinfo.get("patches"),
        "split_info": pinfo.get("split"),
        "config": cfg.to_dict(),
        "history": history,
    }
    with open(os.path.join(out_dir, f"history_{cfg.model}.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return model, summary


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=list(MODELS), default=None)
    ap.add_argument("--patch-dir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--weight-decay", type=float, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument(
        "--device",
        default=None,
        help="cuda, mps or cpu; detected automatically if unset",
    )
    args = ap.parse_args()

    over = {
        k: v
        for k, v in {
            "model": args.model,
            "patch_dir": args.patch_dir,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "patience": args.patience,
            "seed": args.seed,
        }.items()
        if v is not None
    }
    if args.no_augment:
        over["augment"] = False

    cfg = Config().override(**over)
    device = pick_device(args.device)
    out_dir = args.out or cfg.model_dir

    train(cfg, device, out_dir)


if __name__ == "__main__":
    main()
