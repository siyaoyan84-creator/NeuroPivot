"""Bounded real checks: frozen scoring and training-side gradients, never training."""
import argparse
import contextlib
import copy
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode=True
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
from release_check_utils import safe_load,snapshot,dump,read_trial,checkpoint_metadata


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['preflight','frozen','gradient','both'],default='preflight')
    for name in ['affectgpt-root','model-root','cfg-path','checkpoint','task-checkpoint','residual-checkpoint','output-dir']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--fold',type=int,choices=range(1,6),required=True)
    p.add_argument('--anchor',type=int,choices=[1,2],required=True)
    for prefix in ['eval','grad']:
        p.add_argument('--'+prefix+'-subject',type=int)
        p.add_argument('--'+prefix+'-av-npz',type=Path)
        p.add_argument('--'+prefix+'-physiology-npz',type=Path)
    p.add_argument('--eval-trials',type=int,nargs='+',default=[0])
    p.add_argument('--grad-trial',type=int,default=0)
    p.add_argument('--check-task-gradient',action='store_true')
    p.add_argument('--trusted-checkpoints',action='store_true',help='Acknowledge these are user-trusted assets; weights_only stays mandatory')
    p.add_argument('--device',default='cuda',help='Use the user/scheduler selected CUDA device; no fixed GPU index')
    return p


def checkpoint_contract(args,task,cp,report):
    import numpy as np
    from neuropivot.training.protocol import build_split
    expected=build_split(args.fold,'outer')
    for name,value in [('task',task),('residual',cp)]:
        for key,want in [('train_subjects',expected['train']),('outer_test_subjects',expected['test']),('validation_subject',2),('seed',42)]:
            if key not in value or value[key]!=want:raise ValueError(name+' missing/conflicting critical field: '+key)
    if task.get('emotion_names')!=['neutral','anger','happiness','sadness','calmness']:raise ValueError('Missing/conflicting five-class task label mapping')
    if cp.get('feature_dim')!=66 or cp.get('representation')!='va_pivot' or cp.get('mode')!='real':raise ValueError('Not a main-method residual checkpoint')
    if tuple(task['task_token'].shape)!=(1,1,3584):raise ValueError('Invalid task-token shape')
    mean=np.asarray(cp['feature_mean']);std=np.asarray(cp['feature_std'])
    if mean.shape!=(66,) or std.shape!=(66,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std<=1e-6).any():raise ValueError('Invalid saved train-only statistics; never refit them')
    provenance=cp.get('anchor_contract')
    if provenance is None:report['provenance_unknown'].append('Residual checkpoint does not record its fitted anchor; existing weights only validate computations, not the final historical chain')
    elif provenance.get('refed_anchor_test_subject')!=args.anchor or provenance.get('seed')!=42:raise ValueError('Residual anchor identity conflicts')
    for name,value,keys in [('task',task,['checkpoint','base_checkpoint','av_cache']),('residual',cp,['physiology_cache','base_checkpoint'])]:
        for key in keys:
            if key not in value:report['provenance_unknown'].append(name+' checkpoint does not record '+key)
    report['saved_statistics']={'shape':[66],'fitted_now':False,'source':str(args.residual_checkpoint)}
    return mean,std


@contextlib.contextmanager
def observed_pretrained(report):
    """Observe real HF load diagnostics without replacing any model computation."""
    import torch
    from transformers import PreTrainedModel
    original=PreTrainedModel.from_pretrained.__func__;original_load=torch.load
    def audited(cls,*a,**kw):
        requested=kw.get('output_loading_info',False);kw['output_loading_info']=True
        model,info=original(cls,*a,**kw)
        entry={'class':cls.__name__,'path':str(a[0] if a else kw.get('pretrained_model_name_or_path')),'missing_keys':info.get('missing_keys',[]),'unexpected_keys':info.get('unexpected_keys',[]),'mismatched_keys':info.get('mismatched_keys',[]),'error_msgs':info.get('error_msgs',[])}
        report['foundation_loads'].append(entry)
        # Regenerated position-ID buffers are the only blanket missing-key allowance.
        bad=[k for k in entry['missing_keys'] if not k.endswith('position_ids')]
        if bad or entry['unexpected_keys'] or entry['mismatched_keys'] or entry['error_msgs']:raise RuntimeError('Foundation loading discrepancy; inspect foundation_loads, do not silently accept it')
        return (model,info) if requested else model
    def guarded_load(*a,**kw):
        if kw.get('weights_only') is False:raise RuntimeError('Unsafe pickle load requested during check; refused')
        kw['weights_only']=True
        return original_load(*a,**kw)
    PreTrainedModel.from_pretrained=classmethod(audited);torch.load=guarded_load
    try:yield
    finally:PreTrainedModel.from_pretrained=classmethod(original);torch.load=original_load


def initialize_model(args,report):
    import torch
    from neuropivot.integrations.affectgpt.backend import Config,registry,validate_model
    from types import SimpleNamespace
    from omegaconf import OmegaConf
    cfg=Config(SimpleNamespace(cfg_path=str(args.cfg_path),options=[]))
    if cfg.model_cfg.get('ckpt') or cfg.model_cfg.get('ckpt_2') or cfg.model_cfg.get('ckpt_3'):raise ValueError('Check YAML must leave ckpt/ckpt_2/ckpt_3 empty; pass the one explicit --checkpoint')
    report['effective_model_config']=OmegaConf.to_container(cfg.model_cfg,resolve=True)
    report['effective_model_config']['ckpt_3']=str(args.checkpoint)
    # Same production constructor/from_config, then the same one partial-state load,
    # explicitly observed instead of the original strict=False diagnostics being discarded.
    report["model_initialization_attempts"]=1
    with observed_pretrained(report):model=registry.get_model_class(cfg.model_cfg.arch).from_config(cfg.model_cfg)
    report["model_initializations"]=1
    state=safe_load(args.checkpoint,args.trusted_checkpoints)
    if not isinstance(state,dict) or not isinstance(state.get('model'),dict):raise ValueError('AffectGPT checkpoint must contain model state mapping')
    result=model.load_state_dict(state['model'],strict=False)
    missing=list(result.missing_keys);extra=list(result.unexpected_keys)
    new=[k for k in missing if k.startswith('physio_projector.')]
    external=[k for k in missing if k not in new and k.startswith(('llama_model.','visual_encoder.','acoustic_encoder.')) and 'lora_' not in k]
    critical=[k for k in missing if k not in new+external]
    report['affectgpt_state_dict']={'loaded_path':str(args.checkpoint),'missing_keys':missing,'unexpected_keys':extra,'allowed_new_physio':new,'external_pretrained_missing_in_partial_checkpoint':external,'critical_missing_keys':critical,'partial_checkpoint_key_count':len(state['model'])}
    # All LoRA and projection/fusion state must be supplied; only frozen pretrained
    # tensors loaded above and the unused new 2D physio branch may be absent.
    if critical or extra:raise RuntimeError('Critical AffectGPT partial checkpoint mismatch')
    if not report['foundation_loads']:raise RuntimeError('No observed real foundation-model loads')
    validate_model(model);del state;gc.collect()
    for p in model.parameters():p.requires_grad_(False);p.grad=None
    model.eval();model.llama_model=model.llama_model.to(torch.device(args.device)).eval();model.llama_model.config.use_cache=False
    report['special_tokens']={k:int(getattr(model,k+'_PATCH_TOKEN_ID')) for k in ['PHYSIO','MULTI','AUDIO','FRAME']}
    return model


def run_checks(args,report,task,cp,mean,std,rows):
    import numpy as np
    import torch
    from neuropivot.integrations.affectgpt.affect_plumbing import Builder,forward_loss
    from neuropivot.models.residual import ResidualAdapter
    from neuropivot.inference.scoring import predict_emotion
    device=torch.device(args.device)
    if device.type!='cuda' or not torch.cuda.is_available():raise RuntimeError('Real check requires an available user-selected CUDA device; no CPU/double fallback')
    if device.index is not None:torch.cuda.set_device(device)
    torch.manual_seed(42);torch.cuda.manual_seed(42)
    report['cuda']['selected_logical_device']=torch.cuda.current_device()
    report['cuda']['selected_device_name']=torch.cuda.get_device_name()
    report['cuda']['CUDA_VISIBLE_DEVICES']=os.environ.get('CUDA_VISIBLE_DEVICES')
    model=initialize_model(args,report)  # exactly one actual model initialization
    token=task['task_token'].detach().clone().float().to(device);token.requires_grad_(False)
    adapter=ResidualAdapter(66).to(device).eval();loaded=adapter.load_state_dict(cp['adapter'],strict=True)
    report['residual_state_dict']={'missing_keys':list(loaded.missing_keys),'unexpected_keys':list(loaded.unexpected_keys)}
    for p in adapter.parameters():p.requires_grad_(False)
    versions={name:p._version for name,p in model.named_parameters()};adapter_versions={name:p._version for name,p in adapter.named_parameters()};token_version=token._version
    builder=Builder(model,device,token,adapter)
    def batch(row):return {'label':torch.tensor([row['label']]),'z':torch.from_numpy(row['z']).unsqueeze(0),**{k:torch.from_numpy(row[k]).unsqueeze(0) for k in ['audio','frame','multi']}}
    report['frozen_inference']=[]
    if args.mode in {'frozen','both'}:
        for row,identity in rows['eval']:
            with torch.inference_mode():
                embeds,labels,attention,_,delta=builder.build(batch(row),True)
                if not torch.isfinite(embeds).all() or not torch.isfinite(delta).all():raise ValueError('Nonfinite real Builder tensors')
                pred=predict_emotion(model,builder,row)
            report['frozen_inference'].append({'identity':identity,'embedding_shape':list(embeds.shape),'labels_shape':list(labels.shape),'attention_shape':list(attention.shape),'delta_shape':list(delta.shape),'finite':True,'candidate_nll':pred['candidate_nll'].tolist(),'candidate_token_counts':pred['candidate_token_counts'].tolist(),'prediction':pred['emotion'],'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad)+sum(p.numel() for p in adapter.parameters() if p.requires_grad),'parameters_with_grad':sum(p.grad is not None for p in model.parameters())+sum(p.grad is not None for p in adapter.parameters())})
            del embeds,labels,attention,delta
    if args.mode in {'gradient','both'}:
        row,identity=rows['grad'][0]
        for p in adapter.parameters():p.requires_grad_(True);p.grad=None
        loss,ratio=forward_loss(model,builder,batch(row),True)
        if not torch.isfinite(loss):raise ValueError('Nonfinite residual loss')
        loss.backward();grads={k:float(p.grad.float().norm()) if p.grad is not None else None for k,p in adapter.named_parameters()}
        if not all(v is not None and np.isfinite(v) for v in grads.values()) or not any(v>0 for v in grads.values()):raise RuntimeError('No finite nonzero residual gradient')
        if token.grad is not None or any(p.grad is not None for p in model.parameters()):raise RuntimeError('Frozen model/task received gradients')
        report['residual_gradient']={'identity':identity,'loss':float(loss.detach()),'delta_ratio':ratio,'grad_norms':grads,'trainable_parameters':sum(p.numel() for p in adapter.parameters() if p.requires_grad),'frozen_model_parameters_with_grad':0,'task_has_grad':False,'optimizer_step':False,'meaning':'gradient-chain diagnostic using cloned existing trained weights, not training-process reproduction'}
        del loss
        if args.check_task_gradient:
            from neuropivot.training.task_adapter import TaskPromptBuilder
            for p in adapter.parameters():p.requires_grad_(False);p.grad=None
            probe=torch.nn.Parameter(token.detach().clone())
            e,l,a=TaskPromptBuilder(model,device,probe).build(batch(row))
            task_loss=model.llama_model(inputs_embeds=e,attention_mask=a,labels=l,return_dict=True,use_cache=False).loss;task_loss.backward()
            if probe.grad is None or not torch.isfinite(probe.grad).all() or not float(probe.grad.norm())>0:raise RuntimeError('Invalid task-token gradient')
            if any(p.grad is not None for p in model.parameters()) or any(p.grad is not None for p in adapter.parameters()):raise RuntimeError('Frozen Stage-A component received gradients')
            report['task_gradient']={'identity':identity,'gradient_norm':float(probe.grad.norm()),'trainable_parameters':probe.numel(),'frozen_model_adapter_parameters_with_grad':0,'optimizer_step':False,'meaning':'existing-task-token gradient-chain diagnostic only'}
    if any(p._version!=versions[k] for k,p in model.named_parameters()) or any(p._version!=adapter_versions[k] for k,p in adapter.named_parameters()) or token._version!=token_version:raise RuntimeError('An in-memory parameter value was modified')
    report['in_memory_parameter_values_unchanged']=True


def main():
    args=parser().parse_args()
    for k,v in vars(args).items():
        if isinstance(v,Path):setattr(args,k,v.resolve())
    if 'release_checks' not in args.output_dir.parts:raise ValueError('--output-dir must be inside a separate release_checks directory')
    if len(args.eval_trials) not in [1,2] or len(set(args.eval_trials))!=len(args.eval_trials):raise ValueError('Exactly one or two distinct frozen-scoring trials are supported')
    if args.check_task_gradient and args.mode not in {'gradient','both','preflight'}:raise ValueError('Task gradients require a gradient mode and training-side inputs')
    args.output_dir.mkdir(parents=True,exist_ok=False)
    report={'status':'STARTED','mode':args.mode,'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},'provenance_unknown':[],'foundation_loads':[],'model_initializations':0,'optimizer_steps':0,'statistics_fitted':False,'checkpoint_selection':False,'paper_results_reproduced':False}
    before=None;protected=[];cwd=Path.cwd()
    try:
        for root in [args.affectgpt_root,args.model_root]:
            if args.output_dir.is_relative_to(root):raise ValueError('Check output must not be inside an existing source/model asset directory')
        os.environ['AFFECTGPT_ROOT']=str(args.affectgpt_root);os.environ['AFFECTGPT_MODEL_ROOT']=str(args.model_root);os.environ['NEUROPIVOT_OUTER_FOLD']=str(args.fold)
        os.environ['PYTHONDONTWRITEBYTECODE']='1'
        for key in ['HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_DATASETS_OFFLINE','HF_HUB_DISABLE_TELEMETRY']:os.environ[key]='1'
        for key,sub in [('HF_HOME','hf'),('TORCH_HOME','torch'),('XDG_CACHE_HOME','xdg'),('TRITON_CACHE_DIR','triton'),('CUDA_CACHE_PATH','cuda')]:os.environ[key]=str(args.output_dir/'.cache'/sub)
        import numpy as np
        import torch
        import yaml
        from neuropivot.integrations.affectgpt.inspect import inspect,resolve_checkout
        source=resolve_checkout(args.affectgpt_root);args.affectgpt_root=source
        if args.output_dir.is_relative_to(source):raise ValueError('Output cannot be inside external checkout')
        yaml_cfg=yaml.safe_load(args.cfg_path.read_text(encoding='utf-8'))
        if any(yaml_cfg['model'].get(k) for k in ['ckpt','ckpt_2','ckpt_3']):raise ValueError('Only the explicit checkpoint is allowed, not hidden YAML checkpoints')
        report['configuration']=yaml_cfg
        report['resolved_source']=str(source)
        report['dependencies']={}
        for name in ['torch','numpy','transformers','peft','decord','omegaconf','timm','torchaudio','torchvision']:
            try:report['dependencies'][name]=importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:report['dependencies'][name]='MISSING'
        report['cuda']={'available':torch.cuda.is_available(),'device_count':torch.cuda.device_count()}
        inspection=inspect(source,args.model_root);report['source_files_missing']=inspection['missing_required']+inspection['missing_local_imports'];report['model_asset_inventory']=inspection['model_assets']
        if report['source_files_missing']:raise ValueError('Required modified source missing')
        for name,entry in inspection['model_assets'].items():
            if not entry['config'] or not entry['weight_files'] or entry.get('missing_weight_shards'):raise ValueError('Incomplete model files: '+name)
        task=safe_load(args.task_checkpoint,args.trusted_checkpoints);cp=safe_load(args.residual_checkpoint,args.trusted_checkpoints)
        mean,std=checkpoint_contract(args,task,cp,report)
        expected_adapter={'fc1.weight':(128,66),'fc2.weight':(3584,128)}
        if not torch.isfinite(task['task_token']).all() or any(not torch.isfinite(v).all() for v in cp['adapter'].values()):raise ValueError('Nonfinite task/residual checkpoint tensors')
        if set(cp['adapter'])!=set(expected_adapter) or any(tuple(cp['adapter'][k].shape)!=shape for k,shape in expected_adapter.items()):raise ValueError('Critical residual state key/shape mismatch')
        report['state_dict_load_check']='Actual model missing/unexpected keys are checked only in real modes; preflight checks serialized fields/shapes without constructing a model'
        report['checkpoint_metadata']={'task':checkpoint_metadata(args.task_checkpoint,args.trusted_checkpoints),'residual':checkpoint_metadata(args.residual_checkpoint,args.trusted_checkpoints),'base':checkpoint_metadata(args.checkpoint,args.trusted_checkpoints)}
        rows={'eval':[],'grad':[]}
        for prefix in ['eval','grad']:
            active=args.mode=='preflight' and getattr(args,prefix+'_subject') is not None or args.mode=='both' or args.mode==('frozen' if prefix=='eval' else 'gradient')
            if not active:continue
            subject=getattr(args,prefix+'_subject');av=getattr(args,prefix+'_av_npz');phys=getattr(args,prefix+'_physiology_npz')
            if subject is None or av is None or phys is None:raise ValueError('Explicit '+prefix+' subject/AV/physiology paths required')
            protected.extend([av,phys])
            indices=args.eval_trials if prefix=='eval' else [args.grad_trial]
            for i in indices:
                row,identity=read_trial(av,phys,args.fold,args.anchor,subject,i,gradient=prefix=='grad')
                row['z']=((row.pop('pivot')-mean)/std).astype('float32')
                if not np.isfinite(row['z']).all():raise ValueError('Nonfinite standardized physiology')
                identity['input_shapes']={key:list(row[key].shape) for key in ['audio','frame','multi','z']}
                identity['finite']=True
                rows[prefix].append((row,identity))
        report['trial_contracts']={k:[identity for _,identity in values] for k,values in rows.items()}
        report['gradient_scope']='Only declared training-side cross-fitted physiology; never outer-test or validation'
        protected.extend([args.cfg_path,args.checkpoint,args.task_checkpoint,args.residual_checkpoint])
        protected.extend(source/p for p in inspection['local_import_files'])
        protected.extend(p for p in args.model_root.rglob('*') if p.is_file() and p.suffix in {'.json','.bin','.safetensors','.txt'})
        before=snapshot(protected)
        if args.mode!='preflight':
            with (args.output_dir/'model_loading.log').open('x',encoding='utf-8') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):run_checks(args,report,task,cp,mean,std,rows)
        report['status']='PASS'
        report['historical_final_paper_chain']='Cannot determine from this computational check; inspect saved provenance separately'
    except Exception as exc:
        report['status']='FAIL';report['error']=type(exc).__name__+': '+str(exc)
    finally:
        os.chdir(cwd)
        if before is not None:
            try:
                after=snapshot(protected);report['existing_asset_stat_unchanged']=before==after
                if before!=after:report['status']='FAIL';report['asset_changes']=[k for k in before if before[k]!=after.get(k)]
            except Exception as exc:report['status']='FAIL';report['asset_verification_error']=str(exc)
        report['asset_check_scope']='size,mtime_ns,inode for supplied assets/model files/source; read access time is not compared'
        dump(args.output_dir/'report.json',report)
    print(json.dumps({'status':report['status'],'report':str(args.output_dir/'report.json'),'error':report.get('error'),'provenance_unknown_count':len(report['provenance_unknown']),'paper_chain_verified':False},ensure_ascii=False))
    if report['status']!='PASS':raise SystemExit(2)

if __name__=='__main__':main()
