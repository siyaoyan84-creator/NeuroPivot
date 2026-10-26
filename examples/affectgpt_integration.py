"""Bridge existing modified AffectGPT objects to the exact frozen token builder.

Requires a real model, trained task token, trained adapter, and aligned cached AV.
The builder normalizes no inputs: batch['z'] must use saved training statistics.
"""
from neuropivot.integrations.affectgpt.affect_plumbing import Builder

def build_conditioned_prompt(model,device,task_token,adapter,batch):
    builder=Builder(model,device,task_token,adapter)
    return builder.build(batch,use_residual=True)


def predict_aligned_trial(model,device,task_token,adapter,row):
    from neuropivot.inference.scoring import predict_emotion
    builder=Builder(model,device,task_token,adapter)
    return predict_emotion(model,builder,row)
