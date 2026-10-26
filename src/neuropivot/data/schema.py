from pathlib import Path
import os

"""
Data schemas for manifest files.

Defines the structure of trial-level manifests for EAV and REFED datasets.
"""
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
from pathlib import Path

@dataclass
class EAVTrial:
    """
    Single trial record for EAV dataset.

    Each row represents one complete trial (no windowing).
    """
    dataset: str
    subject_id: str
    subject_number: int
    sequence_id: int
    trial_id: int
    condition: str
    emotion: str
    eeg_path: str
    eeg_variable: str
    eeg_trial_index: int
    eeg_label_path: str
    eeg_label_variable: str
    audio_path: str
    video_path: str
    has_eeg: bool
    has_audio: bool
    has_video: bool
    trial_group_id: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for CSV serialization."""
        return asdict(self)

    @classmethod
    def field_names(cls) -> list:
        """Return field names in order."""
        return ['dataset', 'subject_id', 'subject_number', 'sequence_id', 'trial_id', 'condition', 'emotion', 'eeg_path', 'eeg_variable', 'eeg_trial_index', 'eeg_label_path', 'eeg_label_variable', 'audio_path', 'video_path', 'has_eeg', 'has_audio', 'has_video', 'trial_group_id']

@dataclass
class REFEDTrial:
    """
    Single trial record for REFED dataset.

    Each row represents one subject-video trial.
    """
    dataset: str
    subject_id: str
    subject_number: int
    trial_id: int
    eeg_video_path: str
    eeg_video_variable: str
    eeg_baseline_path: str
    eeg_baseline_variable: str
    fnirs_video_path: str
    fnirs_video_variable: str
    fnirs_baseline_path: str
    fnirs_baseline_variable: str
    label_path: str
    label_variable: str
    eeg_channels: int
    eeg_samples: int
    fnirs_measurements: int
    fnirs_channels: int
    fnirs_samples: int
    label_steps: int
    label_dimensions: int
    trial_group_id: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for CSV serialization."""
        return asdict(self)

    @classmethod
    def field_names(cls) -> list:
        """Return field names in order."""
        return ['dataset', 'subject_id', 'subject_number', 'trial_id', 'eeg_video_path', 'eeg_video_variable', 'eeg_baseline_path', 'eeg_baseline_variable', 'fnirs_video_path', 'fnirs_video_variable', 'fnirs_baseline_path', 'fnirs_baseline_variable', 'label_path', 'label_variable', 'eeg_channels', 'eeg_samples', 'fnirs_measurements', 'fnirs_channels', 'fnirs_samples', 'label_steps', 'label_dimensions', 'trial_group_id']
