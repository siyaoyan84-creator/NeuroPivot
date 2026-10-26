from neuropivot.training.contracts import assert_pivot_provenance
from neuropivot.paths import OUTER_FOLD
from neuropivot.paths import REPO, WORKDIR, fold_text
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from neuropivot.integrations.affectgpt import affect_plumbing as stageb
AV_ROOT = WORKDIR / 'features' / 'gate4' / 'gate4_affectgpt_av_llm_embeddings_v1'
PHYS_ROOT = WORKDIR / 'features' / 'gate4' / fold_text('gate4_eav_grouped_outer01_pivot_v1')

def load_subject(subject, representation):
    av_path = AV_ROOT / f'subject{subject:02d}.npz'
    phys_path = PHYS_ROOT / f'subject{subject:02d}.npz'
    if not av_path.is_file():
        raise RuntimeError(f'Missing A/V cache: {av_path}')
    if not phys_path.is_file():
        raise RuntimeError(f'Missing nested physiology cache: {phys_path}')
    av = np.load(av_path, allow_pickle=True)
    phys = np.load(phys_path, allow_pickle=True)
    assert_pivot_provenance(phys,WORKDIR,OUTER_FOLD)
    required_av = {'subject_number', 'sequence_id', 'label', 'audio_llms', 'frame_llms', 'multi_llms'}
    required_phys = {'subject_number', 'sequence_id', 'label', 'teacher_va', 'pivot_shared'}
    missing_av = required_av - set(av.files)
    missing_phys = required_phys - set(phys.files)
    if missing_av:
        raise RuntimeError(f'subject{subject:02d}: missing A/V keys {sorted(missing_av)}')
    if missing_phys:
        raise RuntimeError(f'subject{subject:02d}: missing physiology keys {sorted(missing_phys)}')
    av_subject = np.asarray(av['subject_number']).astype(np.int64)
    phys_subject = np.asarray(phys['subject_number']).astype(np.int64)
    av_seq = np.asarray(av['sequence_id']).astype(np.int64)
    phys_seq = np.asarray(phys['sequence_id']).astype(np.int64)
    av_label = np.asarray(av['label']).astype(np.int64)
    phys_label = np.asarray(phys['label']).astype(np.int64)
    va = np.asarray(phys['teacher_va']).astype(np.float32)
    pivot = np.asarray(phys['pivot_shared']).astype(np.float32)
    audio = np.asarray(av['audio_llms']).astype(np.float16)
    frame = np.asarray(av['frame_llms']).astype(np.float16)
    multi = np.asarray(av['multi_llms']).astype(np.float16)
    assert av_subject.shape == (100,)
    assert phys_subject.shape == (100,)
    assert np.all(av_subject == subject)
    assert np.all(phys_subject == subject)
    assert av_seq.shape == (100,)
    assert phys_seq.shape == (100,)
    if not np.array_equal(av_seq, phys_seq):
        raise RuntimeError(f'subject{subject:02d}: A/V vs nested physiology sequence_id mismatch')
    if not np.array_equal(av_label, phys_label):
        raise RuntimeError(f'subject{subject:02d}: A/V vs nested physiology label mismatch')
    assert len(np.unique(av_seq)) == 100
    assert set(av_label.tolist()) == {0, 1, 2, 3, 4}
    assert va.shape == (100, 2)
    assert pivot.shape == (100, 64)
    assert audio.shape == (100, 1, 3584)
    assert frame.shape == (100, 1, 3584)
    assert multi.shape == (100, 1, 3584)
    assert np.isfinite(va).all()
    assert np.isfinite(pivot).all()
    feature = np.concatenate([va, pivot], axis=1).astype(np.float32)
    expected_dim = 66
    assert feature.shape == (100, expected_dim)
    assert np.isfinite(feature).all()
    rows = []
    for i in range(100):
        rows.append({'subject': int(subject), 'sequence_id': int(av_seq[i]), 'label': int(av_label[i]), 'feature': feature[i].copy(), 'audio': audio[i].copy(), 'frame': frame[i].copy(), 'multi': multi[i].copy()})
    av.close()
    phys.close()
    return rows

def load_subjects(subjects, representation):
    rows = []
    for subject in subjects:
        rows.extend(load_subject(subject, 'va_pivot'))
    return rows

class CachedDataset(Dataset):

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        return {'subject': r['subject'], 'sequence_id': r['sequence_id'], 'label': r['label'], 'z': torch.from_numpy(r['z']).float(), 'audio': torch.from_numpy(r['audio']), 'frame': torch.from_numpy(r['frame']), 'multi': torch.from_numpy(r['multi'])}
