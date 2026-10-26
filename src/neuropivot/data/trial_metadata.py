import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from neuropivot.config import load_config
EMOTIONS={"neutral":0,"anger":1,"happiness":2,"sadness":3,"calmness":4}

def main():
    parser=argparse.ArgumentParser(description="Rebuild exact speaking-trial identity/media metadata; no physiology predictions")
    parser.add_argument("--paths-config", type=Path, required=True)
    args=parser.parse_args()
    cfg=load_config(args.paths_config)
    frame=pd.read_csv(cfg.manifest_dir/"eav_manifest.csv")
    out=cfg.feature_dir/"gate4"/"eav_trial_metadata_v1"
    out.mkdir(parents=True,exist_ok=True)
    for subject in range(2,43):
        rows=frame[(frame.subject_number==subject)&(frame.condition.str.lower()=="speaking")].sort_values("eeg_trial_index").reset_index(drop=True)
        if len(rows)!=100 or rows.sequence_id.nunique()!=100:
            raise ValueError(f"subject{subject:02d}: expected 100 distinct speaking trials")
        labels=rows.emotion.str.lower().map(EMOTIONS)
        if labels.isna().any():
            raise ValueError("Unknown emotion")
        np.savez_compressed(out/f"subject{subject:02d}.npz",subject_number=np.full(100,subject,dtype=np.int64),sequence_id=rows.sequence_id.to_numpy(dtype=np.int64),parent_trial_group_id=rows.trial_group_id.to_numpy(dtype=str),label=labels.to_numpy(dtype=np.int64),audio_path=rows.audio_path.to_numpy(dtype=str),video_path=rows.video_path.to_numpy(dtype=str))
