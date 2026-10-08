# External validation on IDRiD (Indian Diabetic Retinopathy Image Dataset, CC BY 4.0).
# The model was trained only on APTOS 2019, so every IDRiD image is unseen data from
# a different hospital, population and camera.
#   python -m scripts.evaluate_external --root "data/B. Disease Grading" --out reports/idrid.json
#
# Download "B. Disease Grading.zip" from https://ieee-dataport.org/node/956 and unzip it.
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score

from src.config import SAVE_PATH
from src.predict import inference_transforms, load_model

SPLITS = {
    "train": ("1. Original Images/a. Training Set", "2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv"),
    "test": ("1. Original Images/b. Testing Set", "2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv"),
}
N_BOOTSTRAP = 2000


def load_split(root, split):
    img_dir, csv = SPLITS[split]
    # the training CSV has trailing empty columns
    df = pd.read_csv(os.path.join(root, csv)).iloc[:, :2]
    df.columns = ["image", "grade"]
    df["path"] = df["image"].map(lambda i: os.path.join(root, img_dir, f"{i}.jpg"))
    df["split"] = split
    return df


def metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    ref_y, ref_p = y >= 2, p >= 2  # referable DR: moderate or worse
    return {
        "n": int(len(y)),
        "accuracy": accuracy_score(y, p),
        "quadratic_weighted_kappa": cohen_kappa_score(y, p, weights="quadratic"),
        "macro_f1": f1_score(y, p, average="macro", labels=list(range(5)), zero_division=0),
        "referable_sensitivity": float((ref_p & ref_y).sum() / max(ref_y.sum(), 1)),
        "referable_specificity": float((~ref_p & ~ref_y).sum() / max((~ref_y).sum(), 1)),
    }


def bootstrap_ci(y, p, seed=0):
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    draws = {k: [] for k in ("accuracy", "quadratic_weighted_kappa", "referable_sensitivity", "referable_specificity")}
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, len(y), len(y))
        m = metrics(y[idx], p[idx])
        for k in draws:
            draws[k].append(m[k])
    return {k: [round(float(np.percentile(v, 2.5)), 3), round(float(np.percentile(v, 97.5)), 3)] for k, v in draws.items()}


def report(name, df):
    m = metrics(df["grade"], df["pred"])
    out = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}
    out["ci95"] = bootstrap_ci(df["grade"], df["pred"])
    out["confusion_matrix"] = confusion_matrix(df["grade"], df["pred"], labels=list(range(5))).tolist()
    out["true_grade_counts"] = df["grade"].value_counts().sort_index().to_dict()
    return {name: out}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=os.path.join("data", "B. Disease Grading"))
    parser.add_argument("--out", default=None, help="Optional path to write metrics JSON")
    parser.add_argument("--predictions", default=None, help="Optional path to write per-image predictions CSV")
    args = parser.parse_args()

    df = pd.concat([load_split(args.root, s) for s in SPLITS], ignore_index=True)
    model, device = load_model(SAVE_PATH)

    preds, confs = [], []
    with torch.no_grad():
        for path in df["path"]:
            x = inference_transforms(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
            prob = torch.softmax(model(x), dim=1)[0].cpu()
            preds.append(int(prob.argmax()))
            confs.append(round(float(prob.max()), 4))
    df["pred"], df["confidence"] = preds, confs

    results = {"dataset": "IDRiD disease grading (external, never used in training)"}
    results.update(report("all_516", df))
    results.update(report("official_test_103", df[df["split"] == "test"]))

    print(json.dumps(results, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
    if args.predictions:
        df.drop(columns="path").to_csv(args.predictions, index=False)


if __name__ == "__main__":
    main()
