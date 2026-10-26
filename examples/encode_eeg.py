"""Extract real EEG representations from preprocessed windows using trained weights."""
import argparse
from pathlib import Path
import numpy as np
import torch
from neuropivot.inference.interfaces import EEGEncoder
p=argparse.ArgumentParser()
p.add_argument("--checkpoint",required=True)
p.add_argument("--eeg",required=True,help="Preprocessed EAV float32 .npy [B,30,500]")
p.add_argument("--output",required=True)
a=p.parse_args()
encoder=EEGEncoder.from_checkpoint(a.checkpoint)
result=encoder.encode_eeg(torch.from_numpy(np.load(a.eeg,allow_pickle=False)))
np.savez(a.output,**{k:v.cpu().numpy() for k,v in result.items()})
