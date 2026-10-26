"""Read-only helpers shared by the two server checks. No unsafe pickle fallback."""
import hashlib
import json
import math
from pathlib import Path
import zipfile

META_KEYS={'subject_number','teacher_eav_test_subject','teacher_grouped_job_id','teacher_checkpoint','upstream_protocol','upstream_seed','upstream_refed_anchor_test_subject','upstream_fold_id','upstream_outer_fold','upstream_split_role','upstream_inner_fold','outer_test_subjects','permanently_forbidden_from_upstream_train_val','nested_train_subjects','nested_val_subjects','nested_test_subjects','pivot_dim','pivot_l2_normalized','metadata_source','metadata_fields_used'}
SOURCE_WORDS=('source','checkpoint','cache','path','anchor','protocol','subject','fold','seed','mode','config','provenance','representation','condition','emotion','fit_role')


def safe_load(path, trusted, meta=False):
    if not trusted:raise ValueError('Checkpoint inspection requires --trusted-checkpoints for user-owned trusted assets')
    import torch
    try:return torch.load(path,map_location='meta' if meta else 'cpu',weights_only=True,**({'mmap':True} if meta else {}))
    except Exception as exc:raise RuntimeError('Safe weights_only checkpoint read failed; no unsafe fallback: '+str(exc)) from exc


def scalar(value):
    import numpy as np
    a=np.asarray(value)
    if a.size!=1:raise ValueError('Expected scalar metadata')
    return a.item()


def json_value(value, limit=80):
    if value is None or isinstance(value,(str,bool,int,float)):return value
    if isinstance(value,Path):return str(value)
    if isinstance(value,dict):return {str(k):json_value(v,limit) for k,v in value.items() if any(w in str(k).lower() for w in SOURCE_WORDS)}
    if isinstance(value,(tuple,list)):
        return [json_value(v,limit) for v in value] if len(value)<=limit else {'count':len(value),'values':'omitted'}
    if hasattr(value,'shape'):return {'shape':list(value.shape),'dtype':str(value.dtype),'values':'omitted'}
    return {'type':type(value).__name__,'value':'omitted'}


def checkpoint_metadata(path,trusted):
    value=safe_load(path,trusted,meta=True)
    if not isinstance(value,dict):raise ValueError('Checkpoint root must be a mapping')
    selected={k:json_value(v) for k,v in value.items() if any(w in k.lower() for w in SOURCE_WORDS) and k not in {'model','model_state','adapter'}}
    return {'path':str(Path(path).resolve()),'keys':sorted(value),'metadata':selected,'tensor_payloads':'omitted; map_location=meta, weights_only=True','source_claims':'self-reported, not proof of the final paper chain'}


def npz_metadata(path):
    import numpy as np
    path=Path(path);arrays={};metadata={}
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if not name.endswith('.npy'):continue
            with z.open(name) as stream:
                version=np.lib.format.read_magic(stream)
                if version==(1,0):shape,order,dtype=np.lib.format.read_array_header_1_0(stream)
                elif version==(2,0):shape,order,dtype=np.lib.format.read_array_header_2_0(stream)
                else:raise ValueError('Unsupported NPY header version; do not load with pickle')
            arrays[name[:-4]]={'shape':list(shape),'dtype':str(dtype)}
    with np.load(path,allow_pickle=False) as data:
        for key in META_KEYS & set(data.files):
            spec=arrays[key];count=math.prod(spec['shape'])
            if count>10000 or count*np.dtype(spec['dtype']).itemsize>2_000_000:metadata[key]={'value':'omitted, too large'};continue
            a=data[key]
            if a.dtype.kind=='O':raise ValueError('Object metadata requires pickle and is not accepted')
            if key in {'subject_number','teacher_eav_test_subject','teacher_checkpoint'}:metadata[key]=np.unique(a).tolist()
            elif a.size==1:metadata[key]=a.item()
            elif a.size<=80:metadata[key]=a.tolist()
            else:metadata[key]={'count':int(a.size),'value':'omitted'}
    return {'path':str(path.resolve()),'arrays':arrays,'metadata':metadata,'source_claims':'self-reported cache metadata; compare with authentic checkpoint/launch records'}


def expected_role(fold,subject):
    from neuropivot.training.protocol import build_split,job_id_for
    outer=build_split(fold,'outer')
    if subject in outer['test']:role='outer'
    elif subject in outer['train']:
        role=next(r for r in ['inner1','inner2','inner3','inner4','inner5'] if subject in build_split(fold,r)['test'])
    else:raise ValueError('Subject is excluded/validation, not a supported release-check subject')
    return role,job_id_for(fold,role),build_split(fold,role)


def read_trial(av_path,phys_path,fold,anchor,subject,index,gradient=False):
    import numpy as np
    from neuropivot.training.protocol import build_split
    role,job,split=expected_role(fold,subject)
    if gradient and role=='outer':raise ValueError('Gradient checks cannot use outer-test subjects')
    with np.load(av_path,allow_pickle=False) as av,np.load(phys_path,allow_pickle=False) as phys:
        for key in ['subject_number','sequence_id','label']:
            if key not in av or key not in phys:raise ValueError('Missing aligned identity: '+key)
        if index<0 or index>=len(phys['label']):raise ValueError('Trial index out of range')
        for key in ['subject_number','sequence_id','label']:
            if av[key][index]!=phys[key][index]:raise ValueError('AV/physiology mismatch: '+key)
        if int(phys['subject_number'][index])!=subject:raise ValueError('Requested subject does not match cache')
        checks={'upstream_outer_fold':fold,'upstream_refed_anchor_test_subject':anchor,'upstream_seed':42,'upstream_split_role':role,'teacher_grouped_job_id':job}
        for key,want in checks.items():
            if key not in phys or scalar(phys[key])!=want:raise ValueError('Missing/conflicting physiology provenance: '+key)
        for key,want in [('nested_train_subjects',split['train']),('nested_val_subjects',split['val']),('nested_test_subjects',split['test'])]:
            if key not in phys or phys[key].tolist()!=want:raise ValueError('Missing/conflicting cross-fit split: '+key)
        if gradient and subject in split['train']+split['val']:raise ValueError('Subject leaked into its physiology teacher fitting')
        row={'label':int(phys['label'][index]),'subject':subject,'sequence_id':int(phys['sequence_id'][index])}
        if row['label'] not in range(5):raise ValueError('Invalid class ID')
        for key in ['audio','frame','multi']:
            field=key if key in av else key+'_llms';value=np.asarray(av[field][index])
            if value.shape!=(1,3584) or not np.isfinite(value).all():raise ValueError('Invalid real cached AV '+key)
            row[key]=value.copy()
        va=phys['teacher_va'][index];latent=phys['pivot_shared'][index]
        if va.shape!=(2,) or latent.shape!=(64,):raise ValueError('Expected VA2D and latent64D')
        row['pivot']=np.concatenate([va,latent]).astype('float32')
        if row['pivot'].shape!=(66,) or not np.isfinite(row['pivot']).all():raise ValueError('Invalid 66D physiology')
    return row,{'subject':subject,'trial_index':index,'sequence_id':row['sequence_id'],'role':role,'job_id':job,'nested_train_subjects':split['train'],'nested_test_subjects':split['test'],'validation_subjects':split['val'],'cache_proof_scope':'saved cross-fit identity fields checked; historical generation bytes not re-derived'}


def snapshot(paths):
    result={}
    for path in sorted(set(Path(p).resolve() for p in paths)):
        if not path.is_file():raise FileNotFoundError(path)
        s=path.stat();result[str(path)]={'bytes':s.st_size,'mtime_ns':s.st_mtime_ns,'inode':s.st_ino}
    return result


def _report_json_default(value):
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=repr)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(
        f"Unsupported report value: {type(value).__name__}"
    )


def dump(path,value):
    # Serialize before creating the file to avoid partial JSON reports.
    payload = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
        default=_report_json_default,
    )
    with Path(path).open('x', encoding='utf-8') as f:
        f.write(payload + '\n')
