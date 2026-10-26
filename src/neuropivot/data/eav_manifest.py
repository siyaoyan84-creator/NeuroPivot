from pathlib import Path
import os

"""
EAV manifest builder.

Generates trial-level manifest for EAV dataset.
Each row represents one complete trial (no windowing).
"""
from pathlib import Path
from typing import List, Dict, Any
import json
from .schema import EAVTrial
from .mat_header import read_mat_header, MATHeaderError

class EAVManifestError(Exception):
    """Exception raised when EAV manifest building fails."""
    pass

def _find_data_variable_by_shape(mat_path: Path, expected_shape: tuple) -> str:
    """
    Find MAT variable by expected shape.

    Args:
        mat_path: Path to MAT file
        expected_shape: Expected shape tuple

    Returns:
        Variable name

    Raises:
        EAVManifestError: If no unique variable with expected shape found
    """
    try:
        header = read_mat_header(mat_path)
    except MATHeaderError as e:
        raise EAVManifestError(f'Failed to read MAT header {mat_path}: {e}')
    candidates = [var for var, shape, _ in header if tuple(shape) == expected_shape]
    if len(candidates) == 0:
        raise EAVManifestError(f'{mat_path.name}: No variable with shape {expected_shape} found')
    elif len(candidates) > 1:
        raise EAVManifestError(f'{mat_path.name}: Multiple variables with shape {expected_shape}: {candidates}')
    return candidates[0]

def _parse_eav_video_filename(filename: str) -> Dict[str, Any]:
    """
    Parse EAV video filename.

    Format: {sequence_id:03d}_Trial_{trial_id:02d}_{Condition}_{Emotion}.mp4
    Example: 001_Trial_01_Listening_Neutral.mp4

    Args:
        filename: Video filename

    Returns:
        Dictionary with sequence_id, trial_id, condition, emotion

    Raises:
        EAVManifestError: If filename format is invalid
    """
    stem = Path(filename).stem
    parts = stem.split('_')
    if len(parts) != 5:
        raise EAVManifestError(f'Invalid video filename format: {filename}')
    try:
        sequence_id = int(parts[0])
        trial_id = int(parts[2])
        condition = parts[3].lower()
        emotion = parts[4].lower()
    except (ValueError, IndexError) as e:
        raise EAVManifestError(f'Failed to parse video filename {filename}: {e}')
    if condition not in ['listening', 'speaking']:
        raise EAVManifestError(f"Invalid condition '{condition}' in {filename}, expected 'listening' or 'speaking'")
    return {'sequence_id': sequence_id, 'trial_id': trial_id, 'condition': condition, 'emotion': emotion}

def _parse_eav_audio_filename(filename: str) -> Dict[str, Any]:
    """
    Parse EAV audio filename.

    Format: {sequence_id:03d}_Trial_{audio_trial:02d}_Speaking_{Emotion}_aud.wav
    Example: 004_Trial_02_Speaking_Anger_aud.wav

    Note: audio_trial token does not necessarily equal video trial_id.
    Pairing is done by sequence_id only.

    Args:
        filename: Audio filename

    Returns:
        Dictionary with sequence_id, audio_trial, condition, emotion

    Raises:
        EAVManifestError: If filename format is invalid
    """
    stem = Path(filename).stem
    parts = stem.split('_')
    if len(parts) != 6:
        raise EAVManifestError(f'Invalid audio filename format: {filename}')
    try:
        sequence_id = int(parts[0])
        audio_trial = int(parts[2])
        condition = parts[3].lower()
        emotion = parts[4].lower()
        aud_suffix = parts[5].lower()
    except (ValueError, IndexError) as e:
        raise EAVManifestError(f'Failed to parse audio filename {filename}: {e}')
    if condition != 'speaking':
        raise EAVManifestError(f"Invalid condition '{condition}' in audio {filename}, expected 'speaking'")
    if aud_suffix != 'aud':
        raise EAVManifestError(f"Invalid suffix '{aud_suffix}' in audio {filename}, expected 'aud'")
    return {'sequence_id': sequence_id, 'audio_trial': audio_trial, 'condition': condition, 'emotion': emotion}

def _build_eav_subject_manifest(subject_dir: Path, subject_id: str, subject_number: int) -> List[EAVTrial]:
    """
    Build manifest for a single EAV subject.

    Args:
        subject_dir: Subject directory (e.g., subject1/)
        subject_id: Subject identifier (e.g., "subject1")
        subject_number: Subject number (e.g., 1)

    Returns:
        List of EAVTrial records for this subject

    Raises:
        EAVManifestError: If subject data is invalid
    """
    trials = []
    eeg_dir = subject_dir / 'EEG'
    audio_dir = subject_dir / 'Audio'
    video_dir = subject_dir / 'Video'
    if not eeg_dir.exists():
        raise EAVManifestError(f'{subject_id}: EEG directory not found')
    if not audio_dir.exists():
        raise EAVManifestError(f'{subject_id}: Audio directory not found')
    if not video_dir.exists():
        raise EAVManifestError(f'{subject_id}: Video directory not found')
    eeg_file = eeg_dir / f'{subject_id}_eeg.mat'
    label_file = eeg_dir / f'{subject_id}_eeg_label.mat'
    if not eeg_file.exists():
        raise EAVManifestError(f'{subject_id}: EEG file not found: {eeg_file.name}')
    if not label_file.exists():
        raise EAVManifestError(f'{subject_id}: Label file not found: {label_file.name}')
    try:
        eeg_variable = _find_data_variable_by_shape(eeg_file, (10000, 30, 200))
        label_variable = _find_data_variable_by_shape(label_file, (10, 200))
    except EAVManifestError as e:
        raise EAVManifestError(f'{subject_id}: {e}')
    audio_map = {}
    audio_files = sorted(audio_dir.glob('*.wav'))
    for audio_file in audio_files:
        try:
            parsed_audio = _parse_eav_audio_filename(audio_file.name)
            seq_id = parsed_audio['sequence_id']
            audio_emotion = parsed_audio['emotion']
            if seq_id in audio_map:
                raise EAVManifestError(f'{subject_id}: Duplicate audio sequence_id {seq_id}: {audio_map[seq_id][0].name} and {audio_file.name}')
            audio_map[seq_id] = (audio_file, audio_emotion)
        except EAVManifestError as e:
            raise EAVManifestError(f'{subject_id}: {e}')
    video_files = sorted(video_dir.glob('*.mp4'))
    if len(video_files) == 0:
        raise EAVManifestError(f'{subject_id}: No video files found')
    for video_file in video_files:
        parsed = _parse_eav_video_filename(video_file.name)
        sequence_id = parsed['sequence_id']
        trial_id = parsed['trial_id']
        condition = parsed['condition']
        emotion = parsed['emotion']
        has_audio = False
        audio_path_str = ''
        if sequence_id in audio_map:
            audio_file, audio_emotion = audio_map[sequence_id]
            if audio_emotion != emotion:
                raise EAVManifestError(f'{subject_id}: Emotion mismatch for sequence {sequence_id}: video={emotion}, audio={audio_emotion}')
            has_audio = True
            audio_path_str = str(audio_file)
        if condition == 'speaking' and (not has_audio):
            raise EAVManifestError(f'{subject_id}: Speaking trial sequence {sequence_id} missing audio file')
        trial = EAVTrial(dataset='eav', subject_id=subject_id, subject_number=subject_number, sequence_id=sequence_id, trial_id=trial_id, condition=condition, emotion=emotion, eeg_path=str(eeg_file), eeg_variable=eeg_variable, eeg_trial_index=sequence_id - 1, eeg_label_path=str(label_file), eeg_label_variable=label_variable, audio_path=audio_path_str, video_path=str(video_file), has_eeg=True, has_audio=has_audio, has_video=True, trial_group_id=f'eav:{subject_id}:{sequence_id:03d}')
        trials.append(trial)
    return trials

def build_eav_manifest(eav_root: Path) -> List[EAVTrial]:
    """
    Build EAV manifest from root directory.

    Args:
        eav_root: Root directory containing subject1/, subject2/, ..., subject42/

    Returns:
        List of EAVTrial records

    Raises:
        EAVManifestError: If manifest building fails
    """
    trials = []
    subject_dirs = [d for d in eav_root.iterdir() if d.is_dir() and d.name.startswith('subject')]
    if len(subject_dirs) == 0:
        raise EAVManifestError('No subject directories found')

    def extract_subject_number(subject_dir: Path) -> int:
        try:
            return int(subject_dir.name.replace('subject', ''))
        except ValueError:
            raise EAVManifestError(f'Invalid subject directory name: {subject_dir.name}')
    subject_dirs = sorted(subject_dirs, key=extract_subject_number)
    for subject_dir in subject_dirs:
        subject_id = subject_dir.name
        subject_number = extract_subject_number(subject_dir)
        subject_trials = _build_eav_subject_manifest(subject_dir, subject_id, subject_number)
        trials.extend(subject_trials)
    group_ids = [t.trial_group_id for t in trials]
    if len(set(group_ids)) != len(trials):
        raise EAVManifestError(f'Duplicate trial_group_ids found: {len(trials)} trials, {len(set(group_ids))} unique IDs')
    return trials

def validate_eav_manifest_dataset(trials: List[EAVTrial]) -> None:
    """
    Validate EAV manifest represents complete dataset.

    Args:
        trials: List of EAVTrial records

    Raises:
        EAVManifestError: If validation fails
    """
    subjects = set((t.subject_id for t in trials))
    if len(subjects) != 42:
        raise EAVManifestError(f'Expected 42 subjects, found {len(subjects)}')
    if len(trials) != 8400:
        raise EAVManifestError(f'Expected 8400 trials, found {len(trials)}')
    from collections import Counter
    trials_per_subject = Counter((t.subject_id for t in trials))
    for subject_id, count in trials_per_subject.items():
        if count != 200:
            raise EAVManifestError(f'{subject_id}: Expected 200 trials, found {count}')

def generate_eav_summary(trials: List[EAVTrial]) -> Dict[str, Any]:
    """
    Generate summary statistics for EAV manifest.

    Args:
        trials: List of EAVTrial records

    Returns:
        Dictionary with summary statistics
    """
    from collections import Counter
    summary = {'total_trials': len(trials), 'subjects': len(set((t.subject_id for t in trials))), 'trials_per_subject': len(trials) // len(set((t.subject_id for t in trials))) if trials else 0, 'emotion_distribution': dict(Counter((t.emotion for t in trials))), 'condition_distribution': dict(Counter((t.condition for t in trials))), 'eeg_variable_distribution': dict(Counter((t.eeg_variable for t in trials))), 'trials_with_audio': sum((1 for t in trials if t.has_audio)), 'trials_with_video': sum((1 for t in trials if t.has_video)), 'trials_with_eeg': sum((1 for t in trials if t.has_eeg)), 'unique_trial_group_ids': len(set((t.trial_group_id for t in trials)))}
    return summary
