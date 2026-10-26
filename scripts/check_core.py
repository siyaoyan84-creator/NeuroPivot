"""Real CPU PyTorch API checks; no AffectGPT, datasets or pretrained weights."""
import argparse
import ast
import importlib
import json
import sys
import tempfile
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original-root',type=Path,help='Optional explicit original project for exact implementation comparison')
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    sys.path[:]=[x for x in sys.path if Path(x or ".").resolve()!=Path(__file__).resolve().parent]
    if sys.version_info<(3,10):raise RuntimeError('Python >=3.10 required')
    import numpy as np
    import torch
    from torch import nn
    from neuropivot.models.ordinal import Gate3EEGPivotV3
    from neuropivot.models.residual import ResidualAdapter
    from neuropivot.inference.interfaces import EEGEncoder,AVConditioner
    from neuropivot.data.normalization import Normalization
    torch.set_num_threads(2);torch.manual_seed(42)
    model=Gate3EEGPivotV3().eval()
    reference=None;old_adapter_class=None
    if args.original_root:
        original=args.original_root.resolve();sys.path.insert(0,str(original))
        reference=importlib.import_module('src.models.gate3_eeg_pivot_v3').Gate3EEGPivotV3().eval()
        reference.load_state_dict(model.state_dict(),strict=True)
        source=original/'scripts/train_gate4_grouped_outer01_three_conditions_label_fix_v1.py'
        node=next(n for n in ast.parse(source.read_text(encoding='utf-8-sig')).body if isinstance(n,ast.ClassDef) and n.name=='ResidualAdapter')
        namespace={'torch':torch,'nn':nn};exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),namespace)
        old_adapter_class=namespace['ResidualAdapter']
        sys.path.remove(str(original))
    result={'python':sys.version.split()[0],'torch':torch.__version__,'device':'cpu','skipped':0,'original_comparison':reference is not None,'eeg':{}}
    with tempfile.TemporaryDirectory() as tmp:
        tmp=Path(tmp)
        for dataset,c,t in [('refed',64,200),('eav',30,500)]:
            x=torch.randn(2,c,t);mean=np.zeros((1,c,1),dtype='float32');std=np.ones_like(mean)
            encoder=EEGEncoder(model,mean,std,dataset)
            out=encoder.encode_eeg(x)
            assert [list(out[k].shape) for k in ['va','latent','pivot']]==[[2,2],[2,64],[2,66]]
            assert all(torch.isfinite(v).all() for v in out.values())
            assert torch.equal(out['pivot'],torch.cat([out['va'],out['latent']],1))
            assert ((out['va']>=0)&(out['va']<=1)).all()
            path=tmp/(dataset+'.pt')
            torch.save({'model_state':model.state_dict(),'public_format_version':1,'normalization':{'mean':torch.from_numpy(mean),'std':torch.from_numpy(std),'dataset':dataset}},path)
            restored=EEGEncoder.from_checkpoint(path,dataset)
            error=float((restored.encode_eeg(x)['pivot']-out['pivot']).abs().max())
            assert error==0
            item={'va':[2,2],'latent':[2,64],'pivot':[2,66],'checkpoint_max_abs_error':error}
            if reference is not None:
                with torch.no_grad():
                    old=getattr(reference,'encode_'+dataset)(x);new=getattr(model,'encode_'+dataset)(x)
                errors={k:float((new[k]-old[k]).abs().max()) for k in old}
                assert all(e==0 for e in errors.values());item['original_max_abs_error']=errors
            result['eeg'][dataset]=item
        rng=np.random.default_rng(42);raw=rng.normal(size=(8,66)).astype('float32')
        n=Normalization.fit(raw,'pivot',[3]*8,[3,4]);path=tmp/'normalization.npz';n.save(path);saved=Normalization.load(path)
        assert np.array_equal(n.transform(raw),saved.transform(raw));result['normalization_max_abs_error']=float(np.max(np.abs(n.transform(raw)-saved.transform(raw))))
        adapter=ResidualAdapter(66);z=torch.randn(2,66)
        assert torch.equal(adapter(z),torch.zeros(2,3584))
        adapter(z).square().sum().backward();adapter.zero_grad(set_to_none=True)
        # Nonzero output weights are essential to exercise the first-layer gradient too.
        with torch.no_grad():adapter.fc2.weight.normal_(0,.01)
        av=nn.Parameter(torch.randn(2,1,3584),requires_grad=False)
        task=nn.Parameter(torch.zeros(1,1,3584),requires_grad=False)
        for param in model.parameters():param.requires_grad_(False)
        out=av+adapter(z).unsqueeze(1)
        loss=out.square().mean();loss.backward()
        gradients={k:float(v.grad.norm()) if v.grad is not None else None for k,v in adapter.named_parameters()}
        assert all(v is not None and v>0 for v in gradients.values())
        assert all(v.grad is None for v in model.parameters()) and av.grad is None and task.grad is None
        mean=np.linspace(-.2,.2,66,dtype='float32');std=np.linspace(.5,1.5,66,dtype='float32')
        conditioner=AVConditioner(adapter,mean,std);pivot=torch.randn(2,66)
        with torch.no_grad():expected=av+adapter((pivot-torch.from_numpy(mean))/torch.from_numpy(std)).unsqueeze(1)
        actual=conditioner.condition_av(av,pivot);delta=actual-av
        relation_error=float((actual-expected).abs().max());assert relation_error==0 and float(delta.norm())>0
        path=tmp/'residual.pt';torch.save({'adapter':adapter.state_dict(),'feature_dim':66,'representation':'va_pivot','mode':'real','feature_mean':torch.from_numpy(mean),'feature_std':torch.from_numpy(std)},path)
        restored=AVConditioner.from_checkpoint(path);roundtrip=float((restored.condition_av(av,pivot)-actual).abs().max());assert roundtrip==0
        result['residual']={'trainable_parameters':sum(p.numel() for p in adapter.parameters() if p.requires_grad),'grad_norms':gradients,'frozen_eeg_parameters_with_grad':sum(p.grad is not None for p in model.parameters()),'frozen_av_grad':av.grad is not None,'frozen_task_grad':task.grad is not None,'nonzero_delta_norm':float(delta.norm()),'addition_max_abs_error':relation_error,'checkpoint_max_abs_error':roundtrip}
        assert result['residual']['trainable_parameters']==467200
        if old_adapter_class:
            old=old_adapter_class(66).eval();old.load_state_dict(adapter.state_dict(),strict=True)
            with torch.no_grad():error=float((old(z)-adapter(z)).abs().max())
            assert error==0;result['residual']['original_max_abs_error']=error
    assert not any(k=='my_affectgpt' or k.startswith('my_affectgpt.') for k in sys.modules)
    result['affectgpt_imported']=False;result['status']='PASS'
    text=json.dumps(result,indent=2)
    if args.output:args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(text+'\n',encoding='utf-8')
    print(text)

if __name__=='__main__':main()
