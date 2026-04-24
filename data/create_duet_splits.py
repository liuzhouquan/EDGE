"""
Create music_id-level train/val/test splits for the AIST++ duet task.

Why a new split?
----------------
The official AIST++ crossmodal splits have val and test sharing the SAME 10
music IDs (data leakage). For the duet task we must split at the music_id
level (all sequences from the same song go to the same partition), otherwise
sequences from the same song appear in both train and eval.

Split design (60 music IDs total, 6 per genre × 10 genres)
-----------------------------------------------------------
  Test (10 IDs, 1/genre) : the 10 IDs used in the official crossmodal_test.txt
                           — keeps our test set comparable to prior work.
  Val  (10 IDs, 1/genre) : the highest-indexed ID per genre not already in Test
                           — completely disjoint from Train and Test.
  Train(40 IDs, 4/genre) : all remaining IDs.

Output files (written to data/splits/)
---------------------------------------
  duet_train.txt  — sequence names (one per line, no extension)
  duet_val.txt
  duet_test.txt
  duet_split_summary.txt — human-readable summary

Run from the EDGE project root:
    python data/create_duet_splits.py \
        --motions_dir data/edge_aistpp/motions \
        --out_dir     data/splits
"""

import argparse
import os
from collections import defaultdict
from pathlib import Path


# ── Fixed music_id assignments ────────────────────────────────────────────────
# Test: same IDs as the official AIST++ crossmodal_test.txt (enables comparison)
TEST_IDS = {
    "mBR0", "mHO5", "mJB5", "mJS3", "mKR2",
    "mLH4", "mLO2", "mMH3", "mPO1", "mWA0",
}

# Val: one ID per genre, highest-indexed not already in TEST_IDS
VAL_IDS = {
    "mBR5",  # BR: {0→test, 1-4→train, 5→val}
    "mHO4",  # HO: {5→test, 0-3→train, 4→val}
    "mJB4",  # JB: {5→test, 0-3→train, 4→val}
    "mJS5",  # JS: {3→test, 0-2,4→train, 5→val}
    "mKR5",  # KR: {2→test, 0-1,3-4→train, 5→val}
    "mLH5",  # LH: {4→test, 0-3→train, 5→val}
    "mLO5",  # LO: {2→test, 0-1,3-4→train, 5→val}
    "mMH5",  # MH: {3→test, 0-2,4→train, 5→val}
    "mPO5",  # PO: {1→test, 0,2-4→train, 5→val}
    "mWA5",  # WA: {0→test, 1-4→train, 5→val}
}


def parse_music_id(filename: str) -> str | None:
    """Extract music_id from a sequence filename (with or without extension)."""
    base = os.path.splitext(os.path.basename(filename))[0]
    parts = base.split("_")
    return parts[4] if len(parts) >= 5 else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--motions_dir",
        default="data/edge_aistpp/motions",
        help="Directory containing AIST++ .pkl motion files",
    )
    parser.add_argument(
        "--out_dir",
        default="data/splits",
        help="Output directory for split .txt files",
    )
    args = parser.parse_args()

    motions_dir = Path(args.motions_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not motions_dir.is_dir():
        raise FileNotFoundError(f"motions_dir not found: {motions_dir}")

    # ── 1. Group all sequences by music_id ────────────────────────────────────
    by_music: dict[str, list[str]] = defaultdict(list)
    by_music_dancers: dict[str, set[str]] = defaultdict(set)

    for f in sorted(motions_dir.iterdir()):
        if f.suffix != ".pkl":
            continue
        mid = parse_music_id(f.name)
        if mid is None:
            continue
        seq_name = f.stem  # filename without extension
        by_music[mid].append(seq_name)
        dancer_id = f.stem.split("_")[3]
        by_music_dancers[mid].add(dancer_id)

    all_music_ids = sorted(by_music.keys())
    assert all(mid in TEST_IDS | VAL_IDS | set(all_music_ids) for mid in TEST_IDS | VAL_IDS), \
        "TEST_IDS or VAL_IDS contain music IDs not present in motions_dir"

    # ── 2. Assign sequences to splits ─────────────────────────────────────────
    train_seqs, val_seqs, test_seqs = [], [], []

    for mid in all_music_ids:
        seqs = by_music[mid]
        if mid in TEST_IDS:
            test_seqs.extend(seqs)
        elif mid in VAL_IDS:
            val_seqs.extend(seqs)
        else:
            train_seqs.extend(seqs)

    # ── 3. Write split files ───────────────────────────────────────────────────
    splits = {
        "duet_train": train_seqs,
        "duet_val":   val_seqs,
        "duet_test":  test_seqs,
    }
    for name, seqs in splits.items():
        out_path = out_dir / f"{name}.txt"
        with open(out_path, "w") as f:
            f.write("\n".join(sorted(seqs)) + "\n")
        print(f"Wrote {len(seqs):4d} sequences → {out_path}")

    # ── 4. Write human-readable summary ───────────────────────────────────────
    summary_lines = [
        "AIST++ Duet Split Summary",
        "=" * 60,
        "",
        "Design: split at music_id level (no same-song data across splits)",
        f"  Train: 40 music_ids (4/genre × 10 genres)",
        f"  Val  : 10 music_ids (1/genre × 10 genres) — disjoint from Test",
        f"  Test : 10 music_ids (1/genre × 10 genres) — same as official crossmodal_test",
        "",
        f"  Train sequences: {len(train_seqs)}",
        f"  Val   sequences: {len(val_seqs)}",
        f"  Test  sequences: {len(test_seqs)}",
        f"  Total          : {len(train_seqs) + len(val_seqs) + len(test_seqs)}",
        "",
        "Per-genre music_id assignments:",
        "-" * 60,
    ]

    genres = sorted({mid[1:3] for mid in all_music_ids})
    for genre in genres:
        genre_ids = sorted(mid for mid in all_music_ids if mid[1:3] == genre)
        train_g  = [m for m in genre_ids if m not in TEST_IDS and m not in VAL_IDS]
        val_g    = [m for m in genre_ids if m in VAL_IDS]
        test_g   = [m for m in genre_ids if m in TEST_IDS]
        summary_lines.append(
            f"  {genre}  train={train_g}  val={val_g}  test={test_g}"
        )

    summary_lines += [
        "",
        "Duet pairing stats (sequences with ≥2 dancers → valid lead/follower pairs):",
        "-" * 60,
    ]
    for split_name, mid_set in [("train", None), ("val", VAL_IDS), ("test", TEST_IDS)]:
        if split_name == "train":
            mid_set = {m for m in all_music_ids if m not in TEST_IDS and m not in VAL_IDS}
        total_pairs = sum(
            len(by_music_dancers[m]) * (len(by_music_dancers[m]) - 1)
            for m in mid_set if len(by_music_dancers[m]) >= 2
        )
        n_music = len(mid_set)
        summary_lines.append(
            f"  {split_name:5s}: {n_music} music_ids, "
            f"ordered dancer pairs = {total_pairs} "
            f"(× sequences per dancer for actual training pairs)"
        )

    summary_path = out_dir / "duet_split_summary.txt"
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines) + "\n")

    print()
    print("\n".join(summary_lines))
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
