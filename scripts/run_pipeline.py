"""Plan or explicitly execute the actual main chain. Default is print-only."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--anchor',type=int,choices=[1,2],required=True)
    p.add_argument('--folds',type=int,nargs='+',choices=range(1,6),default=[1])
    p.add_argument('--paths-config',type=Path,required=True)
    p.add_argument('--work-dir',type=Path,default=Path('work'))
    p.add_argument('--cfg-path',type=Path,default=Path('configs/affectgpt.main.yaml'))
    p.add_argument('--base-checkpoint',type=Path,required=True)
    p.add_argument('--affectgpt-root',type=Path)
    p.add_argument('--model-root',type=Path)
    p.add_argument('--execute',action='store_true',help='Without this flag no stage is executed')
    args=p.parse_args()
    if sys.version_info<(3,10):raise RuntimeError('Python >=3.10 is required')
    repo=Path(__file__).resolve().parents[1]
    work=args.work_dir.resolve();paths=str(args.paths_config.resolve())
    cfg=str(args.cfg_path.resolve());checkpoint=str(args.base_checkpoint.resolve())
    prefix=[sys.executable,str(repo/'scripts/neuropivot.py'),'--work-dir',str(work)]
    if args.affectgpt_root:prefix+=['--affectgpt-root',str(args.affectgpt_root.resolve())]
    if args.model_root:prefix+=['--model-root',str(args.model_root.resolve())]
    jobs=[('init',[sys.executable,str(repo/'scripts/init_work.py'),'--work-dir',str(work)])]
    def add(stage,*command):jobs.append((stage,prefix+list(command)))
    for dataset in ['refed','eav']:add('manifest-'+dataset,'manifest','--dataset',dataset,'--paths-config',paths)
    add('rounding','refed-rounding')
    for stage,command in [('sample-registry','refed-samples'),('refed-preprocess','preprocess-refed'),('eav-preprocess','preprocess-eav'),('trial-metadata','trial-metadata')]:add(stage,command,'--paths-config',paths)
    add('source','train-refed','--paths-config',paths,'--mode','separate','--refed-test-subject',str(args.anchor),'--seed','42','--run-tag','pilot')
    add('anchor','export-refed-anchor','--paths-config',paths,'--refed-test-subject',str(args.anchor),'--seed','42')
    for index,fold in enumerate(args.folds):
        for role in ['outer','inner1','inner2','inner3','inner4','inner5']:
            add(f'bridge-{fold}-{role}','bridge','--outer-fold',str(fold),'--split-role',role,'--paths-config',paths,'--mode','ordinal_correct','--refed-anchor-test-subject',str(args.anchor),'--seed','42')
        add(f'pivot-{fold}','export-pivots','--outer-fold',str(fold),'--paths-config',paths,'--refed-anchor-test-subject',str(args.anchor))
        if index==0:add('av','cache-av','--subjects','2-42','--cfg-path',cfg,'--checkpoint',checkpoint)
        for stage,command in [('task','train-task'),('residual','train-residual'),('prediction','predict')]:
            rest=['--outer-fold',str(fold),'--cfg-path',cfg,'--checkpoint',checkpoint]
            if stage=='prediction':rest+=['--output',str(work/'predictions'/f'outer{fold:02d}.npz')]
            add(f'{stage}-{fold}',command,*rest)
    if not args.execute:
        print(json.dumps({'executed':False,'anchor_history_confirmed':all(args.anchor==({1:2,2:2,3:2,4:2,5:1}[fold]) for fold in args.folds),'anchor':args.anchor,'folds':args.folds,'stages':[{'stage':name,'argv':argv} for name,argv in jobs]},indent=2))
        return
    for name,argv in jobs:
        print(name,flush=True);subprocess.run(argv,cwd=repo,check=True)

if __name__=='__main__':main()
