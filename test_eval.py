import os
import torch
from torch.utils.data import DataLoader
from model import AudioSpoofDetector, FoRDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_test_set():
    test_dir = os.path.join("dataset", "testing")
    if not os.path.exists(test_dir):
        print("Testing directory not found under dataset/testing")
        return

    test_dataset = FoRDataset(test_dir)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = AudioSpoofDetector().to(DEVICE)
    model.load_state_dict(torch.load("spoof_detector.pth", map_location=DEVICE))
    model.eval()

    correct = 0
    total = 0
    real_correct = 0
    real_total = 0
    fake_correct = 0
    fake_total = 0

    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(features).squeeze(-1)
            preds = (torch.sigmoid(outputs) >= 0.5).float()

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            for p, l in zip(preds, labels):
                if l == 0.0:
                    real_total += 1
                    if p == 0.0: real_correct += 1
                else:
                    fake_total += 1
                    if p == 1.0: fake_correct += 1

    print(f"\n--- Testing Set Evaluation ---")
    print(f"Overall Test Accuracy: {(correct / total) * 100:.2f}%")
    print(f"Real Voice Accuracy:   {(real_correct / real_total) * 100 if real_total > 0 else 0:.2f}% ({real_correct}/{real_total})")
    print(f"Spoof Voice Accuracy:  {(fake_correct / fake_total) * 100 if fake_total > 0 else 0:.2f}% ({fake_correct}/{fake_total})\n")

if __name__ == "__main__":
    evaluate_test_set()