# Evaluate best_model.pth on a labelled split: accuracy, quadratic weighted kappa
# (the APTOS metric), macro F1 and confusion matrix.
#   python -m scripts.evaluate --split test
#   python -m scripts.evaluate --split test --onnx models/dr_efficientnet_b4.onnx
# valid was used for checkpoint selection, so test is the number to report.
import argparse
import json
import os

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from torchvision import transforms

from src.config import BATCH_SIZE, IMAGE_SIZE, SAVE_PATH, TEST_CSV, TEST_IMGS, VAL_CSV, VAL_IMGS
from src.dataset import RetinalDataset, val_transforms
from src.predict import load_model

SPLITS = {
    "valid": (VAL_CSV, VAL_IMGS),
    "test": (TEST_CSV, TEST_IMGS),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=SPLITS, default="test")
    parser.add_argument("--onnx", default=None, help="Evaluate this ONNX export instead of the PyTorch checkpoint")
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
        y_true, y_pred = [], []
        if args.onnx:
            # the ONNX graph normalises internally, so it takes plain 0..1 pixels
            import onnxruntime as ort
            raw = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), transforms.ToTensor()])
            dataset = RetinalDataset(filtered_csv, img_dir, transform=raw)
            session = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
            for image, label in dataset:
                logits, _ = session.run(None, {"image": image.unsqueeze(0).numpy()})
                y_pred.append(int(logits.argmax()))
                y_true.append(int(label))
        else:
            dataset = RetinalDataset(filtered_csv, img_dir, transform=val_transforms)
            loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
            model, device = load_model(SAVE_PATH)
            with torch.no_grad():
                for images, labels in loader:
                    logits = model(images.to(device))
                    y_pred.extend(logits.argmax(dim=1).cpu().tolist())
                    y_true.extend(labels.tolist())
    finally:
        os.remove(filtered_csv)

    metrics = {
        "split": args.split,
        "backend": "onnx" if args.onnx else "pytorch",
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
