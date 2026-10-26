"""Load the locally reviewed modified checkout, offline and on demand."""
import importlib
import importlib.util
import os
import sys
from pathlib import Path
from neuropivot.integrations.affectgpt.inspect import resolve_checkout
_READY=False
_SOURCE=None

def initialize():
    global _READY,_SOURCE
    if _READY:return
    root=resolve_checkout()
    os.environ["AFFECTGPT_ROOT"]=str(root)
    required=['common/config.py','common/registry.py','models/affectgpt.py','models/tokenizer.py','conversation/conversation_video.py']
    missing=[x for x in required if not (root/'my_affectgpt'/x).is_file()]
    if missing:raise ImportError('Incomplete modified AffectGPT: '+', '.join(missing)+'; see docs/NEED_FROM_SERVER.md')
    # Prevent an implicit model/data download. User must prepare assets explicitly.
    os.environ.setdefault('HF_HUB_OFFLINE','1')
    os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
    os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY','1')
    sys.path.insert(0,str(root))
    existing=sys.modules.get('config')
    if existing is not None and Path(getattr(existing,'__file__','')).resolve()!=root/'config.py':
        raise ImportError('A different top-level config module is already loaded; use a fresh process for AffectGPT')
    if existing is None:
        spec=importlib.util.spec_from_file_location('config',root/'config.py')
        module=importlib.util.module_from_spec(spec)
        sys.modules['config']=module
        spec.loader.exec_module(module)
    else:module=existing
    asset_root=Path(os.environ.get('AFFECTGPT_MODEL_ROOT',str(root/'models'))).resolve()
    for field in ['PATH_TO_LLM','PATH_TO_VISUAL','PATH_TO_AUDIO']:
        for key,value in getattr(module,field).items():
            original=Path(value)
            getattr(module,field)[key]=str(asset_root/original.name) if str(original).replace('\\','/').startswith('models/') else str(original if original.is_absolute() else root/original)
    try:
        decord=importlib.import_module('decord');decord.bridge.set_bridge('torch')
        for name in ['tasks','models','runners','processors','datasets.builders']:
            importlib.import_module('my_affectgpt.'+name)
    except ImportError as exc:
        raise ImportError('Modified AffectGPT Python dependencies are incomplete: '+str(exc)+'. See requirements-affectgpt.txt and docs/NEED_FROM_SERVER.md') from exc
    # AffectGPT still contains some relative asset paths; execute its code in its source root.
    # Dataset/model inputs are already absolute; generated NeuroPivot outputs use WORKDIR.
    os.chdir(root)
    _SOURCE=root
    _READY=True

def Config(args):
    initialize()
    cfg=importlib.import_module('my_affectgpt.common.config').Config(args)
    model=cfg.model_cfg
    expected={'arch':'affectgpt','llama_model':'Qwen25','acoustic_encoder':'HUBERT_LARGE','visual_encoder':'CLIP_VIT_LARGE','multi_fusion_type':'attention','video_fusion_type':'attention','audio_fusion_type':'attention'}
    for field,value in expected.items():
        if model.get(field)!=value:raise ValueError('Reviewed main-method config requires '+field+'='+value)
    for field in ['num_audio_query_token','num_video_query_token','num_multi_query_token','num_image_query_token']:
        if model.get(field)!=1:raise ValueError('Reviewed model requires '+field+'=1')
    return cfg

class _Registry:
    def __getattr__(self,name):
        initialize();return getattr(importlib.import_module('my_affectgpt.common.registry').registry,name)
registry=_Registry()

def __getattr__(name):
    routes={'Chat':('my_affectgpt.conversation.conversation_video','Chat'),'MER2025OV_Dataset':('my_affectgpt.datasets.datasets.mer2025ov_dataset','MER2025OV_Dataset'),'BaseProcessor':('my_affectgpt.processors','BaseProcessor')}
    if name not in routes:raise AttributeError(name)
    initialize();module,symbol=routes[name];return getattr(importlib.import_module(module),symbol)


def validate_model(model):
    """Check the real tokenizer/model interface used by the cached-AV main method."""
    vocabulary=model.llama_tokenizer.get_vocab()
    ids=[]
    for name,word in [('MULTI','<MultiHere>'),('AUDIO','<AudioHere>'),('FRAME','<FrameHere>'),('PHYSIO','<PhysioHere>')]:
        token=getattr(model,name+'_PATCH_TOKEN_ID',None)
        if token is None or vocabulary.get(word)!=token:raise ValueError('Modified tokenizer/model token mismatch: '+word)
        ids.append(token)
    if len(set(ids))!=4:raise ValueError('Multimodal special-token IDs must be distinct')
    embeddings=model.llama_model.model.model.embed_tokens
    if int(embeddings.weight.shape[-1])!=3584:raise ValueError('Reviewed NeuroPivot task/residual checkpoints require hidden size 3584')
