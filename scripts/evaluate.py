# Evaluate best_model.pth on a labelled split: accuracy, quadratic weighted kappa
# (the APTOS metric), macro F1 and confusion matrix.
#   python -m scripts.evaluate --split test
# valid was used for checkpoint selection, so test is the number to report.
import argparse
import json
import os

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from src.config import BATCH_SIZE, SAVE_PATH, TEST_CSV, TEST_IMGS, VAL_CSV, VAL_IMGS
from src.dataset import RetinalDataset, val_transforms
from src.predict import load_model

SPLITS = {
    "valid": (VAL_CSV, VAL_IMGS),
    "test": (TEST_CSV, TEST_IMGS),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=SPLITS, default="test")
    parser.add_argument("--out", default=None, help="Optional path to write metrics JSON")
    args = parser.parse_args()

    csv_path, img_dir = SPLITS[args.split]

    # skip rows whose image isn't on disk
    df = pd.read_csv(csv_path)
    present = df["id_code"].apply(lambda i: os.path.exists(os.path.join(img_dir, f"{i}.png")))
    missing = int((~present).sum())
    filtered_csv = csv_path + ".present.csv"
    df[present].to_csv(filtered_csv, index=False)

    try:
        dataset = RetinalDataset(filtered_csv, img_dir, transform=val_transforms)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

        model, device = load_model(SAVE_PATH)
        y_true, y_pred = [], []
        with torch.no_grad():
            for images, labels in loader:
                logits = model(images.to(device))
                y_pred.extend(logits.argmax(dim=1).cpu().tolist())
                y_true.extend(labels.tolist())
    finally:
        os.remove(filtered_csv)

    metrics = {
        "split": args.split,
        "n_images": len(y_true),
        "n_missing_images_skipped": missing,
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "quadratic_weighted_kappa": round(cohen_kappa_score(y_true, y_pred, weights="quadratic"), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(5))).tolist(),
    }

    print(json.dumps(metrics, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
