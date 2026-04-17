# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EDGE (Editable Dance Generation From Music) is a PyTorch implementation of a diffusion-based model for generating physically-plausible, music-synchronized dance motions (CVPR 2023).

**Current development goal:** Extend the model to support **duet/reactive dance generation** — changing the conditioning input from music features alone to (lead dancer motion + music features), so the model learns to generate a follower's motion that is both music-synchronized and responsive to a lead dancer.

## Environment

**Target platform:** Linux x86_64 with NVIDIA GPU, Slurm job scheduler.

```bash
conda env create -f environment-linux.yml
conda activate edge
# then manually install pytorch3d:
pip install pytorch3d \
  --extra-index-url https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu118_pyt210/download.html
accelerate config          # configure fp16 / single-GPU
bash download_model.sh     # download pretrained checkpoint
```

See `TODOLIST.md` for the full reproduction checklist, and `RunInML.md` for Slurm job submission instructions.

## Common Commands

**Train (submit via Slurm):**
```bash
sbatch run_training.sh
# or directly:
accelerate launch train.py --batch_size 128 --epochs 2000 --feature_type jukebox --learning_rate 0.0002
```

**Inference on custom music:**
```bash
python test.py --music_dir custom_music/
```

**Evaluate (Physical Foot Contact metric):**
```bash
python eval/eval_pfc.py --motion_path eval/motions/
```

**Monitor job:**
```bash
squeue -u $USER
tail -f logs/train-output.log
```

## Architecture

**Entry points:** `train.py` and `test.py` instantiate `EDGE` (in `EDGE.py`), which wires together the diffusion model, optimizer, and distributed training via `accelerate`.

**Core modules:**

- **`EDGE.py`** — Top-level class. `train_loop()` runs multi-GPU training with EMA updates and wandb logging; `render_sample()` runs full inference including music feature extraction, DDIM sampling, and video rendering.

- **`model/diffusion.py`** — `GaussianDiffusion`: cosine noise schedule (1000 timesteps), DDIM sampling, in-painting, long-sequence stitching, and a 4-component weighted loss (reconstruction 0.636×, velocity 2.964×, forward kinematics 0.646×, foot contact 10.942×).

- **`model/model.py`** — `DanceDecoder`: 8-layer transformer with FiLM conditioning on diffusion timestep, cross-attention to music embeddings, and rotary positional embeddings. Input is 151-dimensional motion (root pos 3D + 24 joint rotations as 6D vectors × 24 + 4 contact markers).

- **`vis.py`** — `SMPLSkeleton`: 24-joint SMPL kinematic tree, forward kinematics, foot contact detection (joint indices 7, 8, 10, 11). Also handles matplotlib animation and ffmpeg video rendering.

**Current data flow:**
```
music file → feature extraction (Jukebox 4800-dim or baseline 35-dim)
           → DanceDecoder (cross-attention to music embeddings)
           → GaussianDiffusion (DDIM sampler)
           → SMPLSkeleton FK → rendered video
```

**Planned data flow (duet extension):**
```
lead motion (151-dim) + music features (4800-dim)
  → concatenate or dual cross-attention → cond (4951-dim)
  → DanceDecoder → follower motion
```

## Key Conventions

**Motion representation (151-dim):** root position (3) + joint rotations in 6D format (24×6=144) + foot contacts (4).

**Sequence length:** 150 frames (5 seconds @ 30 FPS). Long sequences use `long_ddim_sample()` with overlapping chunks to prevent discontinuities.

**Classifier-free guidance:** 25% of training samples drop music conditioning (`cond_drop_prob=0.25`); inference uses `guidance_weight=2` to amplify conditioning.

**Checkpoint format:** `{"ema_state_dict", "model_state_dict", "optimizer_state_dict", "normalizer"}`. Always load with EMA weights for inference.

**Feature caching:** Jukebox features can be cached as `.npy` files; use `--use_cached_features` to skip recomputation during iterative testing.

**Audio filenames:** Must be simple (e.g., `song.wav`, not filenames with spaces/special chars). Slices are named `songname_slice{N}.wav` and sorted numerically.

**Normalizer:** Motion statistics (mean/std) are computed from the training set and stored in checkpoints — apply consistently across splits.

**Device compatibility:** `pin_memory` in DataLoader is enabled only when CUDA is available. `torch.cuda.empty_cache()` is guarded by `is_available()`.
