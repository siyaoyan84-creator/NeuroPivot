from pathlib import Path
import os

"""
REFED manifest builder.

Generates trial-level manifest for REFED dataset.
Each row represents one subject-video trial.
"""
from pathlib import Path
from typing import List, Dict, Any
import json
from .schema import REFEDTrial
from .mat_header import read_mat_header, MATHeaderError

class REFEDManifestError(Exception):
    """Exception raised when REFED manifest building fails."""
    pass

def _build_refed_subject_manifest(subject_number: int, data_dir: Path, annotations_dir: Path) -> List[REFEDTrial]:
    """
    Build manifest for a single REFED subject.

    Args:
        subject_number: Subject number (1-32)
        data_dir: Data directory containing subject subdirectory
        annotations_dir: Annotations directory containing label MAT files

    Returns:
        List of REFEDTrial records for this subject

    Raises:
        REFEDManifestError: If subject data is invalid
    """
    trials = []
    subject_id = str(subject_number)
    subject_data_dir = data_dir / subject_id
    if not subject_data_dir.exists():
        raise REFEDManifestError(f'Subject {subject_number}: Data directory not found')
    eeg_video_path = subject_data_dir / 'EEG_videos.mat'
    eeg_baseline_path = subject_data_dir / 'EEG_baselines.mat'
    fnirs_video_path = subject_data_dir / 'fNIRS_videos.mat'
    fnirs_baseline_path = subject_data_dir / 'fNIRS_baselines.mat'
    label_path = annotations_dir / f'{subject_number}_label.mat'
    if not eeg_video_path.exists():
        raise REFEDManifestError(f'Subject {subject_number}: EEG_videos.mat not found')
    if not eeg_baseline_path.exists():
        raise REFEDManifestError(f'Subject {subject_number}: EEG_baselines.mat not found')
    if not label_path.exists():
        raise REFEDManifestError(f'Subject {subject_number}: Label file not found')
    try:
        eeg_video_header = read_mat_header(eeg_video_path)
        eeg_baseline_header = read_mat_header(eeg_baseline_path)
        label_header = read_mat_header(label_path)
        fnirs_video_header = []
        fnirs_baseline_header = []
        if fnirs_video_path.exists():
            fnirs_video_header = read_mat_header(fnirs_video_path)
        if fnirs_baseline_path.exists():
            fnirs_baseline_header = read_mat_header(fnirs_baseline_path)
    except MATHeaderError as e:
        raise REFEDManifestError(f'Subject {subject_number}: Failed to read MAT headers: {e}')
    for trial_id in range(1, 16):
        video_var = f'video_{trial_id}'
        eeg_video_vars = [v for v, s, _ in eeg_video_header if v == video_var]
        if not eeg_video_vars:
            raise REFEDManifestError(f"Subject {subject_number} trial {trial_id}: Variable '{video_var}' not found in EEG_videos.mat")
        eeg_video_shape = [s for v, s, _ in eeg_video_header if v == video_var][0]
        if len(eeg_video_shape) != 2 or eeg_video_shape[0] != 64:
            raise REFEDManifestError(f'Subject {subject_number} trial {trial_id}: EEG video shape {eeg_video_shape} invalid, expected (64, T)')
        eeg_channels = eeg_video_shape[0]
        eeg_samples = eeg_video_shape[1]
        eeg_baseline_vars = [v for v, s, _ in eeg_baseline_header if v == video_var]
        if not eeg_baseline_vars:
            raise REFEDManifestError(f"Subject {subject_number} trial {trial_id}: Variable '{video_var}' not found in EEG_baselines.mat")
        eeg_baseline_shape = [s for v, s, _ in eeg_baseline_header if v == video_var][0]
        if tuple(eeg_baseline_shape) != (64, 5000):
            raise REFEDManifestError(f'Subject {subject_number} trial {trial_id}: EEG baseline shape {eeg_baseline_shape} invalid, expected (64, 5000)')
        label_vars = [v for v, s, _ in label_header if v == video_var]
        if not label_vars:
            raise REFEDManifestError(f"Subject {subject_number} trial {trial_id}: Variable '{video_var}' not found in label file")
        label_shape = [s for v, s, _ in label_header if v == video_var][0]
        if len(label_shape) != 2 or label_shape[1] != 2:
            raise REFEDManifestError(f'Subject {subject_number} trial {trial_id}: Label shape {label_shape} invalid, expected (T, 2)')
        label_steps = label_shape[0]
        label_dimensions = label_shape[1]
        expected_samples = label_steps * 1000
        if eeg_samples != expected_samples:
            raise REFEDManifestError(f'Subject {subject_number} trial {trial_id}: EEG samples ({eeg_samples}) != label_steps * 1000 ({expected_samples})')
        fnirs_channels = 0
        fnirs_samples = 0
        fnirs_measurements = 0
        fnirs_video_variable = ''
        fnirs_baseline_variable = ''
        if fnirs_video_header:
            fnirs_video_vars = [v for v, s, _ in fnirs_video_header if v == video_var]
            if fnirs_video_vars:
                fnirs_video_variable = video_var
                fnirs_video_shape = [s for v, s, _ in fnirs_video_header if v == video_var][0]
                if len(fnirs_video_shape) == 3 and fnirs_video_shape[0] == 6 and (fnirs_video_shape[1] == 51):
                    fnirs_channels = fnirs_video_shape[1]
                    fnirs_samples = fnirs_video_shape[2]
                    fnirs_measurements = fnirs_video_shape[0]
        if fnirs_baseline_header:
            fnirs_baseline_vars = [v for v, s, _ in fnirs_baseline_header if v == video_var]
            if fnirs_baseline_vars:
                fnirs_baseline_variable = video_var
                fnirs_baseline_shape = [s for v, s, _ in fnirs_baseline_header if v == video_var][0]
                if tuple(fnirs_baseline_shape) != (6, 51, 238):
                    raise REFEDManifestError(f'Subject {subject_number} trial {trial_id}: fNIRS baseline shape {fnirs_baseline_shape} invalid, expected (6, 51, 238)')
        trial = REFEDTrial(dataset='refed', subject_id=subject_id, subject_number=subject_number, trial_id=trial_id, eeg_video_path=str(eeg_video_path), eeg_video_variable=video_var, eeg_baseline_path=str(eeg_baseline_path), eeg_baseline_variable=video_var, fnirs_video_path=str(fnirs_video_path) if fnirs_video_path.exists() else '', fnirs_video_variable=fnirs_video_variable, fnirs_baseline_path=str(fnirs_baseline_path) if fnirs_baseline_path.exists() else '', fnirs_baseline_variable=fnirs_baseline_variable, label_path=str(label_path), label_variable=video_var, eeg_channels=eeg_channels, eeg_samples=eeg_samples, fnirs_measurements=fnirs_measurements, fnirs_channels=fnirs_channels, fnirs_samples=fnirs_samples, label_steps=label_steps, label_dimensions=label_dimensions, trial_group_id=f'refed:{subject_id}:{trial_id:02d}')
        trials.append(trial)
    return trials

def build_refed_manifest(refed_root: Path) -> List[REFEDTrial]:
    """
    Build REFED manifest from root directory.

    Args:
        refed_root: Root directory containing data/ and annotations/ subdirectories

    Returns:
        List of REFEDTrial records

    Raises:
        REFEDManifestError: If manifest building fails
    """
    trials = []
    data_dir = refed_root / 'data'
    annotations_dir = refed_root / 'annotations'
    if not data_dir.exists():
        raise REFEDManifestError(f'Data directory not found: {data_dir}')
    if not annotations_dir.exists():
        raise REFEDManifestError(f'Annotations directory not found: {annotations_dir}')
    subject_dirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    if len(subject_dirs) == 0:
        raise REFEDManifestError('No subject directories found')
    subject_numbers = sorted([int(d.name) for d in subject_dirs])
    for subject_number in subject_numbers:
        subject_trials = _build_refed_subject_manifest(subject_number, data_dir, annotations_dir)
        trials.extend(subject_trials)
    group_ids = [t.trial_group_id for t in trials]
    if len(set(group_ids)) != len(trials):
        raise REFEDManifestError(f'Duplicate trial_group_ids found: {len(trials)} trials, {len(set(group_ids))} unique IDs')
    return trials

def validate_refed_manifest_dataset(trials: List[REFEDTrial]) -> None:
    """
    Validate REFED manifest represents complete dataset.

    Args:
        trials: List of REFEDTrial records

    Raises:
        REFEDManifestError: If validation fails
    """
    subjects = set((t.subject_id for t in trials))
    if len(subjects) != 32:
        raise REFEDManifestError(f'Expected 32 subjects, found {len(subjects)}')
    if len(trials) != 480:
        raise REFEDManifestError(f'Expected 480 trials, found {len(trials)}')
    from collections import Counter
    trials_per_subject = Counter((t.subject_id for t in trials))
    for subject_id, count in trials_per_subject.items():
        if count != 15:
            raise REFEDManifestError(f'Subject {subject_id}: Expected 15 trials, found {count}')

def generate_refed_summary(trials: List[REFEDTrial]) -> Dict[str, Any]:
    """
    Generate summary statistics for REFED manifest.

    Args:
        trials: List of REFEDTrial records

    Returns:
        Dictionary with summary statistics
    """
    from collections import Counter
    summary = {'total_trials': len(trials), 'subjects': len(set((t.subject_id for t in trials))), 'trials_per_subject': len(trials) // len(set((t.subject_id for t in trials))) if trials else 0, 'eeg_channels': {'min': min((t.eeg_channels for t in trials)), 'max': max((t.eeg_channels for t in trials)), 'unique_values': sorted(set((t.eeg_channels for t in trials)))}, 'eeg_samples': {'min': min((t.eeg_samples for t in trials)), 'max': max((t.eeg_samples for t in trials))}, 'label_steps': {'min': min((t.label_steps for t in trials)), 'max': max((t.label_steps for t in trials))}, 'fnirs_available': sum((1 for t in trials if t.fnirs_video_path)), 'unique_trial_group_ids': len(set((t.trial_group_id for t in trials)))}
    return summary
