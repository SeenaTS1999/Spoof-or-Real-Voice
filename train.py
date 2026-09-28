import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import AudioSpoofDetector, FoRDataset

BATCH_SIZE = 32
LEARNING_RATE = 2.5e-4
EPOCHS = 15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_epoch(model, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for features, labels in dataloader:
        features = features.to(DEVICE)
        labels = labels.to(DEVICE)

        # Mild feature jittering during training to reduce overfitting on microphone noise
        if model.training:
            noise = torch.randn_like(features) * 0.02
            features = features + noise

        optimizer.zero_grad()
        outputs = model(features).squeeze(-1)
        
        # Label smoothing prevents overconfidence on edge cases
        smoothed_labels = labels * 0.9 + 0.05
        loss = criterion(outputs, smoothed_labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * features.size(0)
        preds = (outputs >= 0.0).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, (correct / total) * 100.0


def validate(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(features).squeeze(-1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * features.size(0)
            preds = (outputs >= 0.0).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, (correct / total) * 100.0


def main():
    print(f"Using compute device: {DEVICE}")

    train_dir = os.path.join("dataset", "training")
    val_dir = os.path.join("dataset", "validation")

    train_dataset = FoRDataset(train_dir)
    val_dataset = FoRDataset(val_dir)

    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = AudioSpoofDetector().to(DEVICE)

    # Balanced weight
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_loss = float("inf")

    print("\n--- Training Model with Noise Regularization ---")
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate(model, val_loader, criterion)

        scheduler.step()

        print(
            f"Epoch [{epoch:02d}/{EPOCHS:02d}] | "
            f"Train Loss: {train_loss:.4f} - Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} - Acc: {val_acc:.2f}%"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "spoof_detector.pth")
            print(f"  --> Saved Checkpoint (Val Loss: {val_loss:.4f})")

    print("\nTraining complete.")


if __name__ == "__main__":
    main()