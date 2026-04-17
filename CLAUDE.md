# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

EDGE (Editable Dance Generation From Music) is a PyTorch implementation of a diffusion-based model for generating physically-plausible, music-synchronized dance motions (CVPR 2023).

## Setup

```bash
pip install -r requirements.txt
accelerate config          # configure fp16 / multi-GPU settings
bash download_model.sh     # download pretrained checkpoint from Google Drive
```

## Common Commands

**Train:**
```bash
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

**Convert SMPL output to FBX (requires Blender):**
```bash
python SMPL-to-FBX/Convert.py --input_dir SMPL-to-FBX/smpl_samples/ --output_dir SMPL-to-FBX/fbx_out
```

There is no unit test suite — interactive testing is done via `demo.ipynb`.

## Architecture

**Entry points:** `train.py` and `test.py` instantiate `EDGE` (in `EDGE.py`), which wires together the diffusion model, optimizer, and distributed training via `accelerate`.

**Core modules:**

- **`EDGE.py`** — Top-level class. `train_loop()` runs multi-GPU training with EMA updates and wandb logging; `render_sample()` runs full inference including music feature extraction, DDIM sampling, and video rendering.

- **`model/diffusion.py`** — `GaussianDiffusion`: cosine noise schedule (1000 timesteps), DDIM sampling, in-painting, long-sequence stitching, and a 4-component weighted loss (reconstruction 0.636×, velocity 2.964×, forward kinematics 0.646×, foot contact 10.942×).

- **`model/model.py`** — `DanceDecoder`: 8-layer transformer with FiLM conditioning on diffusion timestep, cross-attention to music embeddings, and rotary positional embeddings. Input is 151-dimensional motion (root pos 3D + 24 joint rotations as 6D vectors × 24 + 4 contact markers).

- **`vis.py`** — `SMPLSkeleton`: 24-joint SMPL kinematic tree, forward kinematics, foot contact detection (joint indices 7, 8, 10, 11). Also handles matplotlib animation and ffmpeg video rendering.

**Data flow:**
```
music file → feature extraction (Jukebox 4800-dim or baseline 35-dim)
          → DanceDecoder (denoising transformer)
          → GaussianDiffusion (DDIM sampler)
          → SMPLSkeleton FK → rendered video
```

## Key Conventions

**Motion representation (151-dim):** root position (3) + joint rotations in 6D format (24×6=144) + foot contacts (4).

**Sequence length:** 150 frames (5 seconds @ 30 FPS). Long sequences use `long_ddim_sample()` with overlapping chunks to prevent discontinuities.

**Classifier-free guidance:** 25% of training samples drop music conditioning (`cond_drop_prob=0.25`); inference uses `guidance_weight=2` to amplify conditioning.

**Checkpoint format:** `{"ema_state_dict", "model_state_dict", "optimizer_state_dict", "normalizer"}`. Always load with EMA weights for inference.

**Feature caching:** Jukebox features can be cached as `.npy` files; use `--use_cached_features` to skip recomputation during iterative testing.

**Audio filenames:** Must be simple (e.g., `song.wav`, not filenames with spaces/special chars). Slices are named `songname_slice{N}.wav` and sorted numerically.

**Normalizer:** Motion statistics (mean/std) are computed from the training set and stored in checkpoints — apply consistently across splits.
