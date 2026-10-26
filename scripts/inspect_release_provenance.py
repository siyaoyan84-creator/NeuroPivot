"""Read only recorded lineage: never infer the final paper chain from folder names."""
import argparse
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from release_check_utils import npz_metadata,checkpoint_metadata,json_value,snapshot,dump


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ['pivot','checkpoint','evaluation','record']:p.add_argument('--'+flag,type=Path,nargs='+',default=[])
    p.add_argument('--fold',type=int,choices=range(1,6))
    p.add_argument('--anchor',type=int,choices=[1,2])
    p.add_argument('--trusted-checkpoints',action='store_true')
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args();out=args.output_dir.resolve()
    if 'release_checks' not in out.parts:raise ValueError('Output must be inside a separate release_checks directory')
    files=[path.resolve() for group in [args.pivot,args.checkpoint,args.evaluation,args.record] for path in group]
    if not files:raise ValueError('Provide explicit assets; no directory/tree scan is performed')
    if out in files:raise ValueError('Output cannot replace an asset')
    out.mkdir(parents=True,exist_ok=False)
    before=snapshot(files);entries=[]
    for kind,paths in [('pivot',args.pivot),('checkpoint',args.checkpoint),('evaluation',args.evaluation),('record',args.record)]:
        for path in paths:
            item={'kind':kind,'path':str(path.resolve()),'evidence':'recorded self-report only'}
            try:
                if path.suffix.lower()=='.npz':item['recorded']=npz_metadata(path)
                elif path.suffix.lower()=='.json':
                    if path.stat().st_size>10_000_000:raise ValueError('JSON exceeds bounded metadata inspection size')
                    item['recorded']={'metadata':json_value(json.loads(path.read_text(encoding='utf-8')))}
                elif kind=='checkpoint':item['recorded']=checkpoint_metadata(path,args.trusted_checkpoints)
                else:raise ValueError('Expected JSON/NPZ or an explicitly trusted checkpoint')
                meta=item['recorded'].get('metadata',{})
                item['missing_source_fields']=[key for key in ['base_checkpoint','task_checkpoint','residual_checkpoint','av_cache','physiology_cache'] if key not in meta]
                if kind=='pivot':
                    item['consistency_warnings']=[]
                    for key,want in [('upstream_outer_fold',args.fold),('upstream_refed_anchor_test_subject',args.anchor)]:
                        if want is not None and meta.get(key)!=want:item['consistency_warnings'].append(key+' absent/conflicting; do not fill it from the directory name')
                    fold=meta.get('upstream_outer_fold');anchor=meta.get('upstream_refed_anchor_test_subject')
                    if fold in [1,2,3,4,5] and anchor is not None and anchor!=({1:2,2:2,3:2,4:2,5:1}[fold]):item['consistency_warnings'].append('Cache anchor metadata conflicts with supplied bridge summaries; those summaries alone do not identify the final cache/weights used')
                item['conclusion']='无法从现有记录确定论文最终使用的完整链路；以上只列出该资产实际保存的来源字段'
            except Exception as exc:item['read_error']=type(exc).__name__+': '+str(exc);item['conclusion']='无法从现有记录确定；未关闭weights_only或allow_pickle安全限制'
            entries.append(item)
    after=snapshot(files)
    result={'status':'READ_ONLY_REPORT','assets_unchanged':before==after,'entries':entries,'paper_chain_verified':False,'namespace_note':'REFED anchor IDs and EAV subject IDs are separate namespaces','minimum_manual_confirmation':['Identify the exact final evaluation file used in each paper table/fold','Supply authentic input checkpoint/cache paths and preferably SHA256 or run IDs for that evaluation','Where absent, confirm task/residual fitted-input/cache identities and actual bridge teacher used; do not invent fields or relabel existing assets']}
    if before!=after:raise RuntimeError('Existing asset metadata changed during read-only inspection')
    dump(out/'provenance.json',result)
    print(json.dumps({'report':str(out/'provenance.json'),'paper_chain_verified':False,'assets_unchanged':True},ensure_ascii=False))

if __name__=='__main__':main()
