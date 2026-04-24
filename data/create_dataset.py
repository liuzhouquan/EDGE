import argparse
import os
from pathlib import Path

from audio_extraction.baseline_features import \
    extract_folder as baseline_extract
from audio_extraction.jukebox_features import extract_folder as jukebox_extract
from filter_split_data import *
from slice import *


def create_dataset(opt):
    if opt.duet:
        # Three-way split (train/val/test) for the duet task
        print("Creating duet train / val / test split")
        split_data_duet(opt.dataset_folder)
        splits_to_process = ["train", "val", "test"]
    else:
        # Original binary split (train/test) for solo mode
        print("Creating train / test split")
        split_data(opt.dataset_folder)
        splits_to_process = ["train", "test"]

    # Slice motions/music into sliding windows
    for split_name in splits_to_process:
        print(f"Slicing {split_name} data")
        slice_aistpp(f"{split_name}/motions", f"{split_name}/wavs")

    # Extract audio features
    for split_name in splits_to_process:
        if opt.extract_baseline:
            print(f"Extracting baseline features ({split_name})")
            baseline_extract(f"{split_name}/wavs_sliced", f"{split_name}/baseline_feats")
        if opt.extract_jukebox:
            print(f"Extracting jukebox features ({split_name})")
            jukebox_extract(f"{split_name}/wavs_sliced", f"{split_name}/jukebox_feats")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stride", type=float, default=0.5)
    parser.add_argument("--length", type=float, default=5.0, help="checkpoint")
    parser.add_argument(
        "--dataset_folder",
        type=str,
        default="edge_aistpp",
        help="folder containing motions and music",
    )
    parser.add_argument("--extract-baseline", action="store_true")
    parser.add_argument("--extract-jukebox", action="store_true")
    parser.add_argument(
        "--duet", action="store_true",
        help="Use duet splits (train/val/test) instead of original solo splits (train/test)",
    )
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    create_dataset(opt)
