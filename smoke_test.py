"""
Minimal smoke test for EDGE — validates train→inference flow on synthetic data.
No real dataset, GPU, or Jukebox features are required.

Usage:
    python smoke_test.py
    python smoke_test.py --epochs 1 --batch_size 2 --n_samples 4
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

os.environ.setdefault("WANDB_MODE", "disabled")

from accelerate import Accelerator, DistributedDataParallelKwargs
from dataset.preprocess import Normalizer
from model.adan import Adan
from model.diffusion import GaussianDiffusion
from model.model import DanceDecoder
from vis import SMPLSkeleton

SEQ_LEN = 150
MOTION_DIM = 151   # 4 contacts + 3 root pos + 24×6 rotations
FEATURE_DIM = 35   # baseline feature dimension (no Jukebox needed)


class SyntheticDataset(Dataset):
    """Random motion + music feature pairs for smoke testing."""

    def __init__(self, n_samples: int = 8, normalizer=None):
        data = torch.randn(n_samples, SEQ_LEN, MOTION_DIM)
        if normalizer is None:
            self.normalizer = Normalizer(data)
        else:
            self.normalizer = normalizer
        self.data = self.normalizer.normalize(data)
        self.n = n_samples

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        x = self.data[idx]
        cond = torch.randn(SEQ_LEN, FEATURE_DIM)
        return x, cond, f"synthetic_{idx}.npy", f"synthetic_{idx}.wav"


def run_smoke_test(epochs: int = 2, batch_size: int = 2, n_samples: int = 8):
    print("=== EDGE Smoke Test ===")

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    device = accelerator.device
    print(f"Device: {device}")

    model = DanceDecoder(
        nfeats=MOTION_DIM,
        seq_len=SEQ_LEN,
        latent_dim=512,
        ff_size=1024,
        num_layers=8,
        num_heads=8,
        dropout=0.1,
        cond_feature_dim=FEATURE_DIM,
        activation=F.gelu,
    )
    smpl = SMPLSkeleton(device)
    diffusion = GaussianDiffusion(
        model,
        SEQ_LEN,
        MOTION_DIM,
        smpl,
        schedule="cosine",
        n_timestep=1000,
        predict_epsilon=False,
        loss_type="l2",
        use_p2=False,
        cond_drop_prob=0.25,
        guidance_weight=2,
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    model = accelerator.prepare(model)
    diffusion = diffusion.to(device)
    optim = Adan(model.parameters(), lr=4e-4, weight_decay=0.02)
    optim = accelerator.prepare(optim)

    train_ds = SyntheticDataset(n_samples=n_samples)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
    )
    train_loader = accelerator.prepare(train_loader)

    # ----- training loop -----
    for epoch in range(1, epochs + 1):
        diffusion.train()
        total = 0.0
        steps = 0
        for x, cond, _, _ in train_loader:
            total_loss, (loss, v_loss, fk_loss, foot_loss) = diffusion(
                x, cond, t_override=None
            )
            optim.zero_grad()
            accelerator.backward(total_loss)
            optim.step()
            total += loss.item()
            steps += 1
        print(f"  Epoch {epoch}/{epochs} — avg recon loss: {total / steps:.4f}")

    # ----- inference (DDIM, no rendering) -----
    diffusion.eval()
    shape = (2, SEQ_LEN, MOTION_DIM)
    sample_cond = torch.randn(2, SEQ_LEN, FEATURE_DIM).to(device)
    with torch.no_grad():
        sample = diffusion.ddim_sample(shape, sample_cond)

    assert sample.shape == torch.Size([2, SEQ_LEN, MOTION_DIM]), (
        f"Unexpected output shape: {sample.shape}"
    )
    print(f"Inference output shape: {list(sample.shape)}  ✓")
    print("=== Smoke Test PASSED ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--n_samples", type=int, default=8)
    args = parser.parse_args()
    run_smoke_test(args.epochs, args.batch_size, args.n_samples)
