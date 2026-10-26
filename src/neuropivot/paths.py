import os
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
WORKDIR = Path(os.environ.get("NEUROPIVOT_WORKDIR",str(REPO/"work"))).resolve()
OUTER_FOLD = int(os.environ.get("NEUROPIVOT_OUTER_FOLD","1"))
from neuropivot.training.protocol import OUTER_FOLDS
TRAIN_SUBJECTS_PUBLIC = sorted(set(range(3,43))-set(OUTER_FOLDS[OUTER_FOLD]))
N_TRAIN = len(TRAIN_SUBJECTS_PUBLIC)
N_TRIALS = N_TRAIN*100

def fold_text(value):
    return value.replace("outer01",f"outer{OUTER_FOLD:02d}").replace("OUTER01",f"OUTER{OUTER_FOLD:02d}").replace("Outer01",f"Outer{OUTER_FOLD:02d}")
