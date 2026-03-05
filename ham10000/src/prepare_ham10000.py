import os
import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# Adjust BASE_DIR if your structure is different
BASE_DIR = Path(__file__).resolve().parents[1]   # goes 2 levels up to HAM10000/
DATA_DIR = BASE_DIR
IMG_DIR = DATA_DIR / "images"
META_CSV = DATA_DIR / "HAM10000_metadata.csv"
SPLIT_DIR = DATA_DIR / "split"

# Train/val/test split ratios
TEST_SIZE = 0.10   # 10% for final test
VAL_SIZE = 0.10    # 10% for validation (from remaining 90%)

def make_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def main():
    # 1) Load metadata
    df = pd.read_csv(META_CSV)

    # Use only image_id and dx (diagnosis label)
    df = df[["image_id", "dx"]].dropna()

    # 2) Add full image path
    df["filename"] = df["image_id"] + ".jpg"
    df["filepath"] = df["filename"].apply(lambda x: IMG_DIR / x)

    # Drop rows where file is missing
    df = df[df["filepath"].apply(lambda p: p.exists())]

    # 3) Stratified split: train + temp (val+test)
    train_df, temp_df = train_test_split(
        df,
        test_size=TEST_SIZE + VAL_SIZE,
        stratify=df["dx"],
        random_state=42,
    )

    # From temp, split into val and test
    relative_val_size = VAL_SIZE / (TEST_SIZE + VAL_SIZE)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=1 - relative_val_size,
        stratify=temp_df["dx"],
        random_state=42,
    )

    print("Train size:", len(train_df))
    print("Val size:  ", len(val_df))
    print("Test size: ", len(test_df))

    # 4) Create folder structure
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        for label in split_df["dx"].unique():
            split_label_dir = SPLIT_DIR / split_name / label
            make_dir(split_label_dir)

        # 5) Copy images into correct folders
        for _, row in split_df.iterrows():
            src = row["filepath"]
            label = row["dx"]
            dst = SPLIT_DIR / split_name / label / src.name
            if not dst.exists():  # avoid duplicate copying
                shutil.copy2(src, dst)

    print("✅ Done! Data organized in:", SPLIT_DIR)

if __name__ == "__main__":
    main()
