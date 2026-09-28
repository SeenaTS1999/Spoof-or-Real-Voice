import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import librosa
import numpy as np

SAMPLE_RATE = 16000
TARGET_DURATION_SEC = 2
TARGET_SAMPLES = SAMPLE_RATE * TARGET_DURATION_SEC
N_MFCC = 30


def preprocess_audio(waveform: torch.Tensor, orig_sr: int) -> torch.Tensor:
    if isinstance(waveform, torch.Tensor):
        audio_np = waveform.squeeze().cpu().numpy()
    else:
        audio_np = waveform

    # 1. Mono conversion
    if audio_np.ndim > 1:
        audio_np = np.mean(audio_np, axis=0)

    # 2. Resample to 16 kHz
    if orig_sr != SAMPLE_RATE:
        audio_np = librosa.resample(audio_np, orig_sr=orig_sr, target_sr=SAMPLE_RATE)

    # 3. Silence Trimming (Removes ambient background room noise before/after speech)
    audio_trimmed, _ = librosa.effects.trim(audio_np, top_db=25)
    if len(audio_trimmed) > SAMPLE_RATE * 0.5:  # Use trimmed audio if valid speech remains
        audio_np = audio_trimmed

    # 4. Peak Normalization
    max_val = np.abs(audio_np).max()
    if max_val > 1e-6:
        audio_np = audio_np / max_val

    # 5. Fixed 2-Second Duration (Pad or Center Crop)
    length = len(audio_np)
    if length < TARGET_SAMPLES:
        pad_size = TARGET_SAMPLES - length
        audio_np = np.pad(audio_np, (0, pad_size), mode='constant')
    else:
        start_idx = (length - TARGET_SAMPLES) // 2
        audio_np = audio_np[start_idx:start_idx + TARGET_SAMPLES]

    # 6. Compute MFCC + Delta + Delta-Delta Features (n_mfcc is lowercase)
    mfcc = librosa.feature.mfcc(y=audio_np, sr=SAMPLE_RATE, n_mfcc=N_MFCC, n_fft=1024, hop_length=256)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    features_np = np.stack([mfcc, delta, delta2], axis=0)

    # Standardize feature map
    mean = features_np.mean(axis=-1, keepdims=True)
    std = features_np.std(axis=-1, keepdims=True) + 1e-6
    features_np = (features_np - mean) / std

    return torch.tensor(features_np, dtype=torch.float32)


class FoRDataset(Dataset):
    def __init__(self, root_dir: str):
        self.file_list = []
        self.labels = []

        real_dir = os.path.join(root_dir, "real")
        fake_dir = os.path.join(root_dir, "fake")

        if os.path.exists(real_dir):
            for fname in os.listdir(real_dir):
                if fname.lower().endswith(('.wav', '.mp3', '.flac')):
                    self.file_list.append(os.path.join(real_dir, fname))
                    self.labels.append(0.0)

        if os.path.exists(fake_dir):
            for fname in os.listdir(fake_dir):
                if fname.lower().endswith(('.wav', '.mp3', '.flac')):
                    self.file_list.append(os.path.join(fake_dir, fname))
                    self.labels.append(1.0)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        label = self.labels[idx]

        try:
            audio_np, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
            features = preprocess_audio(audio_np, sr)
        except Exception:
            features = torch.zeros((3, N_MFCC, 126))

        return features, torch.tensor(label, dtype=torch.float32)


class AudioSpoofDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)