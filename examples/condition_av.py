"""AffectGPT-independent residual application to an existing fused AV embedding."""
import argparse
import numpy as np
import torch
from neuropivot.inference.interfaces import AVConditioner
p=argparse.ArgumentParser()
p.add_argument("--adapter-checkpoint",required=True)
p.add_argument("--av",required=True,help=".npy fused AV [B,1,3584]")
p.add_argument("--pivot",required=True,help=".npy raw aligned physiology [B,66]")
p.add_argument("--output",required=True)
a=p.parse_args()
conditioner=AVConditioner.from_checkpoint(a.adapter_checkpoint)
result=conditioner.condition_av(torch.from_numpy(np.load(a.av)),torch.from_numpy(np.load(a.pivot)))
np.save(a.output,result.cpu().numpy())
