"""Normalization contracts extracted from source channel_stats and final residual fit."""
from pathlib import Path
import numpy as np

class Normalization:
    def __init__(self, mean, std, kind):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.kind = kind
        if self.mean.shape != self.std.shape or not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or np.any(self.std <= 0):
            raise ValueError("Invalid normalization statistics")

    @classmethod
    def fit(cls, values, kind, subjects, allowed_subjects):
        x = np.asarray(values, dtype=np.float32)
        subjects = np.asarray(subjects)
        if len(x) != len(subjects) or not set(subjects.tolist()).issubset(set(allowed_subjects)):
            raise ValueError("Normalization fit contains forbidden subjects")
        if kind == "eeg":
            if x.ndim != 3:
                raise ValueError("EEG must be [N,C,T]")
            mean = x.mean(axis=(0,2))[None,:,None]
            std = np.maximum(x.std(axis=(0,2))[None,:,None], 1e-6)
        elif kind == "pivot":
            if x.ndim != 2 or x.shape[1] != 66:
                raise ValueError("Pivot must be [N,66]")
            mean, std = x.mean(axis=0), x.std(axis=0)
            if float(std.min()) <= 1e-6:
                raise ValueError("Near-constant physiology dimension")
        else:
            raise ValueError(kind)
        return cls(mean, std, kind)

    def transform(self, x):
        return ((np.asarray(x, dtype=np.float32) - self.mean) / self.std).astype(np.float32)

    def save(self, path):
        np.savez(Path(path), mean=self.mean, std=self.std, kind=np.asarray(self.kind), version=np.asarray(1))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            if int(z["version"]) != 1:
                raise ValueError("Unsupported normalization version")
            return cls(z["mean"], z["std"], str(z["kind"]))
