"""Persist stage provenance without changing model/optimization behavior."""
import json
from pathlib import Path

def bind_anchor(cfg, subject, seed=42):
    subject=int(subject)
    if subject not in {1,2}:raise ValueError('Only explicit historical anchor01/02 candidates are supported')
    path=Path(cfg.checkpoint_dir)/'gate3'/'neuropivot_anchor_contract.json'
    contract={'format_version':1,'refed_anchor_test_subject':subject,'seed':int(seed)}
    if path.exists():
        old=json.loads(path.read_text(encoding='utf-8'))
        if old!=contract:raise ValueError('Anchor/seed mismatch in this workspace. Select a separate work-dir and matching paths config; refusing to mix anchor01/02.')
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('x',encoding='utf-8') as f:json.dump(contract,f,indent=2)
    return contract

def assert_checkpoint_anchor(checkpoint, subject):
    provenance=checkpoint.get('anchor_contract')
    if provenance is None:raise ValueError('Legacy bridge checkpoint has no anchor provenance. Supply the original per-job launch/config evidence; do not infer anchor from cache metadata.')
    if provenance.get('refed_anchor_test_subject')!=int(subject) or provenance.get('seed')!=42:
        raise ValueError('Bridge checkpoint anchor provenance does not match requested export')


def assert_pivot_provenance(phys, workdir, outer_fold):
    path=Path(workdir)/'checkpoints/gate3/neuropivot_anchor_contract.json'
    if not path.is_file():raise ValueError('Missing workspace anchor contract; build the explicit anchor first')
    contract=json.loads(path.read_text(encoding='utf-8'))
    for key,want in [('upstream_refed_anchor_test_subject',contract['refed_anchor_test_subject']),('upstream_outer_fold',int(outer_fold)),('upstream_seed',contract['seed'])]:
        if key not in phys.files or int(phys[key])!=want:raise ValueError('Physiology cache provenance mismatch: '+key)
