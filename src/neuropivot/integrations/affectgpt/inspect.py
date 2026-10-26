"""Dependency-light, offline inspection of the actual modified source checkout."""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
import re
from pathlib import Path
from neuropivot.paths import REPO

REQUIRED = {
 "my_affectgpt/common/config.py": ["Config"],
 "my_affectgpt/common/registry.py": ["Registry"],
 "my_affectgpt/models/affectgpt.py": ["AffectGPT", "encode_multi_attention", "from_config"],
 "my_affectgpt/models/tokenizer.py": ["load_tokenizer_from_LLM"],
 "my_affectgpt/conversation/conversation_video.py": ["Chat", "postprocess_audio", "postprocess_frame", "postprocess_multi"],
 "my_affectgpt/datasets/datasets/mer2025ov_dataset.py": ["MER2025OV_Dataset"],
 "my_affectgpt/datasets/datasets/base_dataset.py": ["get_needed_data", "read_frame_face_audio_text"],
 "my_affectgpt/processors/video_processor.py": ["AlproVideoEvalProcessor"],
 "my_affectgpt/processors/blip_processors.py": ["Blip2ImageEvalProcessor"],
 "config.py": [],
}
INFERENCE_YAML="train_configs/mercaptionplus_outputhybird_bestsetup_bestfusion_frame_lz.yaml"

def resolve_checkout(source=None):
    selected=source or os.environ.get("AFFECTGPT_ROOT")
    if not selected:raise FileNotFoundError("Set --affectgpt-root / --source or AFFECTGPT_ROOT explicitly; sibling checkout discovery is disabled")
    base=Path(selected)
    for p in [base,base/"AffectGPT",base/"AffectGPT"/"AffectGPT"]:
        if (p/"my_affectgpt"/"models"/"affectgpt.py").is_file():
            return p.resolve()
    raise FileNotFoundError("Modified AffectGPT checkout not found: set --affectgpt-root or AFFECTGPT_ROOT to the inner directory containing my_affectgpt")

def imports_closure(root):
    pending=list(REQUIRED)+["my_affectgpt/__init__.py","my_affectgpt/runners/__init__.py"]
    found=set();missing=set();external=set()
    def add_module(name):
        if name=="config":
            pending.append("config.py");return
        if name.split('.')[0] not in {"my_affectgpt","toolkit"}:
            external.add(name.split('.')[0]);return
        path=Path(*name.split('.'))
        options=[path.with_suffix('.py'),path/'__init__.py']
        exists=[p for p in options if (root/p).is_file()]
        if not exists:
            # Namespace packages may legitimately lack __init__.py.
            if not (root/path).is_dir():missing.add(str(path).replace('\\','/'))
            return
        for p in exists:pending.append(p.as_posix())
        for parent in path.parents:
            if parent!=Path('.') and (root/parent/'__init__.py').is_file():pending.append((parent/'__init__.py').as_posix())
    while pending:
        rel=pending.pop()
        if rel in found:continue
        path=root/rel
        if not path.is_file():missing.add(rel);continue
        found.add(rel)
        module='.'.join(Path(rel).with_suffix('').parts)
        if module.endswith('.__init__'):module=module[:-9]
        package=module if Path(rel).name=='__init__.py' else module.rpartition('.')[0]
        try:t=ast.parse(path.read_text(encoding='utf-8-sig'))
        except (SyntaxError,UnicodeError) as exc:
            missing.add(rel+': cannot parse '+type(exc).__name__);continue
        for n in ast.walk(t):
            if isinstance(n,ast.Import):
                for alias in n.names:add_module(alias.name)
            elif isinstance(n,ast.ImportFrom):
                name=n.module or ''
                if n.level:
                    try:name=importlib.util.resolve_name('.'*n.level+name,package)
                    except (ImportError,ValueError):continue
                add_module(name)
                for alias in n.names:
                    candidate=name+'.'+alias.name
                    if alias.name!='*' and ((root/Path(*candidate.split('.')).with_suffix('.py')).is_file() or (root/Path(*candidate.split('.'))/'__init__.py').is_file()):add_module(candidate)
    return sorted(found),sorted(missing),sorted(external)

def inspect(source=None,model_root=None):
    root=resolve_checkout(source)
    missing=[];symbols={}
    for rel,names in REQUIRED.items():
        p=root/rel
        if not p.is_file():missing.append(rel);continue
        tree=ast.parse(p.read_text(encoding='utf-8-sig'))
        defs={n.name for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        symbols[rel]={name:name in defs for name in names}
        missing.extend(rel+': '+name for name in names if name not in defs)
    closure,missing_imports,external=imports_closure(root)
    config=(root/'config.py').read_text(encoding='utf-8-sig')
    tokens={}
    for n in ast.parse(config).body:
        if isinstance(n,ast.Assign):
            for target in n.targets:
                if isinstance(target,ast.Name) and target.id.endswith('PATCH_TOKEN') and isinstance(n.value,ast.Constant):tokens[target.id]=n.value.value
    yaml=root/INFERENCE_YAML
    yaml_text=yaml.read_text(encoding='utf-8-sig') if yaml.is_file() else ''
    yaml_defaults=re.findall(r'(?m)^\s*(?:defaults|_base_|include)\s*:',yaml_text)
    model_dir=Path(model_root or os.environ.get('AFFECTGPT_MODEL_ROOT',str(root/'models'))).resolve()
    model_files={}
    for name in ['Qwen2.5-7B-Instruct','clip-vit-large-patch14','chinese-hubert-large']:
        directory=model_dir/name
        weights=list(directory.glob('*.safetensors'))+list(directory.glob('pytorch_model*.bin'))
        entry={'config':(directory/'config.json').is_file(),'weight_files':len(weights),'tokenizer_json':(directory/'tokenizer.json').is_file()}
        if name=='Qwen2.5-7B-Instruct' and (directory/'config.json').is_file():entry['hidden_size']=json.loads((directory/'config.json').read_text()).get('hidden_size')
        if name=='Qwen2.5-7B-Instruct' and (directory/'model.safetensors.index.json').is_file():
            index=json.loads((directory/'model.safetensors.index.json').read_text())
            entry['missing_weight_shards']=sorted(set(index.get('weight_map',{}).values())-{p.name for p in weights})
        model_files[name]=entry
    revision_file=root.parent/'.git/refs/heads/master'
    revision=revision_file.read_text().strip() if revision_file.is_file() else None
    return {'source_root':str(root),'symbol_checks':symbols,'missing_required':missing,'local_import_files':closure,'missing_local_imports':missing_imports,'external_import_names':external,'special_tokens':tokens,'model_arch':'affectgpt' if re.search(r'arch:\s*affectgpt',yaml_text) else None,'inference_yaml_present':yaml.is_file(),'yaml_defaults_directives':yaml_defaults,'default_model_config_reference':{'path':'configs/models/affectgpt.yaml','present':(root/'configs/models/affectgpt.yaml').is_file(),'used_by_public_from_config_route':False},'model_assets':model_files,'recorded_revision':revision,'git_objects_present':(root.parent/'.git/objects').is_dir(),'review_metadata_present':(root/'review_metadata').is_dir() or (root.parent/'review_metadata').is_dir(),'apache_license_present':(root/'LICENSE').is_file(),'lavis_license_present':any((p/'LICENSE_Lavis').is_file() for p in [root,root.parent]),'source_hashes':{rel:hashlib.sha256((root/rel).read_bytes()).hexdigest() for rel in closure}}

def main():
    p=argparse.ArgumentParser(description="Offline modified AffectGPT source/import/token/config inspection; never loads models")
    p.add_argument('--source',type=Path)
    p.add_argument('--model-root',type=Path)
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    result=inspect(args.source,args.model_root)
    if args.output:result["source_root"]="<external-modified-AffectGPT-inner-checkout>"
    text=json.dumps(result,indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(text+'\n',encoding='utf-8')
    print(text)
    if result['missing_required'] or result['missing_local_imports']:raise SystemExit(2)
