from pathlib import Path
import os

import torch
import torch.nn.functional as F
from torch import nn
VA_LEVELS = (0.2, 0.5, 0.8)

def build_va_coordinates():
    coords = []
    for v in VA_LEVELS:
        for a in VA_LEVELS:
            coords.append([v, a])
    return torch.tensor(coords, dtype=torch.float32)

class DatasetSpecificEEGEncoder(nn.Module):

    def __init__(self, channels, embedding_dim=128, dropout=0.25):
        super().__init__()
        self.channels = channels
        self.net = nn.Sequential(nn.Conv1d(channels, 64, kernel_size=15, stride=2, padding=7, bias=False), nn.BatchNorm1d(64), nn.GELU(), nn.Conv1d(64, 96, kernel_size=9, stride=2, padding=4, bias=False), nn.BatchNorm1d(96), nn.GELU(), nn.Conv1d(96, 128, kernel_size=7, stride=2, padding=3, bias=False), nn.BatchNorm1d(128), nn.GELU(), nn.AdaptiveAvgPool1d(1))
        self.out = nn.Sequential(nn.Flatten(), nn.Linear(128, embedding_dim), nn.GELU(), nn.Dropout(dropout))

    def forward(self, x):
        if x.ndim == 4 and x.shape[1] == 1:
            x = x[:, 0]
        if x.ndim != 3:
            raise ValueError('Expected [B,C,T]')
        if x.shape[1] != self.channels:
            raise ValueError(f'Expected {self.channels} channels, got {x.shape[1]}')
        return self.out(self.net(x))

class SharedAffectProjector(nn.Module):

    def __init__(self, input_dim=128, shared_dim=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, shared_dim), nn.LayerNorm(shared_dim), nn.GELU(), nn.Linear(shared_dim, shared_dim))

    def forward(self, x):
        return self.net(x)

class Gate3EEGPivotV2(nn.Module):
    """
    Important separation:

      dataset-specific native task
          !=
      cross-dataset shared affect semantics
    """

    def __init__(self, embedding_dim=128, shared_dim=64, dropout=0.25):
        super().__init__()
        self.eav_encoder = DatasetSpecificEEGEncoder(channels=30, embedding_dim=embedding_dim, dropout=dropout)
        self.refed_encoder = DatasetSpecificEEGEncoder(channels=64, embedding_dim=embedding_dim, dropout=dropout)
        self.eav_classifier = nn.Sequential(nn.Linear(embedding_dim, 64), nn.GELU(), nn.Dropout(dropout), nn.Linear(64, 5))
        self.shared_projector = SharedAffectProjector(input_dim=embedding_dim, shared_dim=shared_dim)
        self.shared_va_head = nn.Sequential(nn.Linear(shared_dim, 32), nn.GELU(), nn.Linear(32, 2), nn.Sigmoid())
        self.register_buffer('va_coordinates', build_va_coordinates())

    def va_to_proto_logits(self, va, sigma=0.15):
        dist2 = ((va[:, None, :] - self.va_coordinates[None, :, :]) ** 2).sum(dim=2)
        return -dist2 / (2.0 * sigma * sigma)

    def encode_eav(self, eeg):
        base = self.eav_encoder(eeg)
        shared = self.shared_projector(base)
        va_pred = self.shared_va_head(shared)
        return {'base': base, 'shared': shared, 'class_logits': self.eav_classifier(base), 'va_pred': va_pred, 'proto_logits': self.va_to_proto_logits(va_pred)}

    def encode_refed(self, eeg):
        base = self.refed_encoder(eeg)
        shared = self.shared_projector(base)
        va_pred = self.shared_va_head(shared)
        return {'base': base, 'shared': shared, 'va_pred': va_pred, 'proto_logits': self.va_to_proto_logits(va_pred)}

    def refed_targets(self, va, sigma=0.15):
        return torch.softmax(self.va_to_proto_logits(va, sigma=sigma), dim=1)

def soft_cross_entropy(logits, target):
    return -(target * F.log_softmax(logits, dim=1)).sum(dim=1).mean()

def soft_target_entropy(target):
    target = target.clamp_min(1e-08)
    return -(target * target.log()).sum(dim=1).mean()

def soft_kl(logits, target):
    ce = soft_cross_entropy(logits, target)
    h = soft_target_entropy(target)
    return (ce - h, ce, h)
