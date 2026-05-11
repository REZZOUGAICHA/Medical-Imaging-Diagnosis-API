import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader

from src.dataset import RetinalDataset, train_transforms, val_transforms
from src.model import build_model
from src.config import (
    TRAIN_CSV, VAL_CSV,
    TRAIN_IMGS, VAL_IMGS,
    BATCH_SIZE, NUM_CLASSES,
    NUM_WORKERS, SAVE_PATH
)


def get_class_weights(csv_path):
    """
    Compute per-class weights to handle class imbalance.
    
    The APTOS dataset is heavily skewed toward class 0 (healthy eyes).
    Without compensation, the model learns to predict class 0 for everything
    and achieves ~49% accuracy while being clinically useless.
    
    Formula: weight = total_samples / (num_classes * samples_in_class)
    Rare classes get higher weights → model penalized more for missing them.
    """
    df = pd.read_csv(csv_path)
    counts = df['diagnosis'].value_counts().sort_index().values
    total = counts.sum()
    weights = total / (len(counts) * counts)
    return torch.FloatTensor(weights)


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Run one full pass over the training set.
    
    The 4 steps repeated for every batch:
    1. zero_grad()     — clear gradients from previous batch
    2. forward pass    — compute predictions
    3. loss.backward() — backpropagation, compute gradients
    4. optimizer.step()— update weights
    
    Returns average loss and accuracy over the epoch.
    """
    model.train()  # enables dropout
    total_loss, correct, total = 0, 0, 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if batch_idx % 10 == 0:
            print(f"  Batch {batch_idx}/{len(loader)} - Loss: {loss.item():.4f}")

    return total_loss / len(loader), correct / total


def validate(model, loader, criterion, device):
    """
    Evaluate model on validation set — no weight updates.
    
    model.eval() disables dropout for consistent, reproducible results.
    torch.no_grad() skips gradient computation — saves memory and speeds
    up inference since we're only predicting, not learning.
    
    Returns average loss and accuracy over the validation set.
    """
    model.eval()  # disables dropout
    total_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(loader), correct / total


def train(num_epochs=10):
    """
    Full training pipeline:
    - Loads datasets and dataloaders
    - Builds EfficientNet-B4 with pretrained ImageNet weights
    - Applies weighted CrossEntropyLoss to handle class imbalance
    - Uses Adam optimizer with ReduceLROnPlateau scheduling
    - Saves best checkpoint based on validation loss
    
    Note: Training was run on Kaggle (H100 GPU).
    To retrain locally, a CUDA-capable GPU is strongly recommended.
    CPU training takes ~90 minutes per epoch.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Data
    train_dataset = RetinalDataset(TRAIN_CSV, TRAIN_IMGS, transform=train_transforms)
    val_dataset   = RetinalDataset(VAL_CSV,   VAL_IMGS,   transform=val_transforms)
    train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                               shuffle=True,  num_workers=NUM_WORKERS)
    val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                               shuffle=False, num_workers=NUM_WORKERS)

    # Model, loss, optimizer
    model         = build_model(num_classes=NUM_CLASSES).to(device)
    class_weights = get_class_weights(TRAIN_CSV).to(device)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)
    optimizer     = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler     = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, mode='min', patience=2)

    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        print(f"\\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 50)

        train_loss, train_acc = train_one_epoch(model, train_loader,
                                                criterion, optimizer, device)
        val_loss,   val_acc   = validate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc*100:.2f}%")

        # Save only when validation loss improves
        # This ensures best_model.pth is always the best generalization
        # checkpoint, not just the most recent one
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), SAVE_PATH)
            print("✓ Best model saved")

    print(f"\\nTraining complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    train()