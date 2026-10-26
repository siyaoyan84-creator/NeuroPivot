"""TEST ONLY. NumPy tensor/model/tokenizer doubles; never imported by production.

These exercise real source control flow and array operations, not torch autograd,
actual transformers, actual processors, CUDA, pretrained weights or paper accuracy.
"""
import ast
import contextlib
import re
from types import SimpleNamespace
import numpy as np

class Tensor(np.ndarray):
    def __new__(cls,value,dtype=None):return np.asarray(value,dtype=dtype).view(cls)
    @property
    def device(self):return 'cpu'
    def to(self,*args,**kwargs):return Tensor(np.asarray(self),kwargs.get('dtype',None))
    def cpu(self):return self
    def numpy(self):return np.asarray(self)
    def detach(self):return self
    def clone(self):return Tensor(np.asarray(self).copy())
    def float(self):return Tensor(self,np.float32)
    def long(self):return Tensor(self,np.int64)
    def unsqueeze(self,dim):return Tensor(np.expand_dims(np.asarray(self),dim))
    def expand(self,*shape):
        shape=tuple(self.shape[i] if n==-1 else n for i,n in enumerate(shape))
        return Tensor(np.broadcast_to(np.asarray(self),shape).copy())
    def numel(self):return np.asarray(self).size
    def size(self,dim=None):return self.shape if dim is None else self.shape[dim]
    def dim(self):return self.ndim
    def norm(self,*args,**kwargs):return Tensor(np.linalg.norm(np.asarray(self)))

class NoGrad(contextlib.ContextDecorator):
    def __enter__(self):return self
    def __exit__(self,*args):return False

class TorchDouble:
    float32=np.float32;float16=np.float16;long=np.int64
    tensor=staticmethod(lambda value,dtype=None,device=None:Tensor(value,dtype))
    from_numpy=staticmethod(lambda value:Tensor(value))
    zeros=staticmethod(lambda *shape,**kw:Tensor(np.zeros(shape,dtype=kw.get('dtype',np.float32))))
    where=staticmethod(lambda value:tuple(Tensor(v) for v in np.where(value)))
    cat=staticmethod(lambda values,dim=0:Tensor(np.concatenate(values,axis=dim)))
    concat=staticmethod(lambda values,axis=0:Tensor(np.concatenate(values,axis=axis)))
    stack=staticmethod(lambda values,dim=0:Tensor(np.stack(values,axis=dim)))
    mean=staticmethod(lambda value,axis=None:Tensor(np.asarray(value).mean(axis=axis)))
    unsqueeze=staticmethod(lambda value,dim:Tensor(np.expand_dims(value,dim)))
    matmul=staticmethod(lambda a,b:Tensor(np.matmul(a,b)))
    isfinite=staticmethod(np.isfinite)
    no_grad=NoGrad;inference_mode=NoGrad
    cuda=SimpleNamespace(amp=SimpleNamespace(autocast=lambda **kwargs:NoGrad()))

torch=TorchDouble()
EMOTIONS=['neutral','anger','happiness','sadness','calmness']

class TokenizerDouble:
    eos_token_id=2;pad_token_id=0;eos_token='EOS'
    special={'<MultiHere>':100,'<AudioHere>':101,'<FrameHere>':102,'<PhysioHere>':103,'<ImageHere>':104,'<FaceHere>':105}
    words={'neutral':[4],'anger':[5,6],'happiness':[7,8,9],'sadness':[10],'calmness':[11,12]}
    def __call__(self,text,**kwargs):
        if text in self.words:return {'input_ids':self.words[text].copy()}
        chunks=re.split('(<(?:Multi|Audio|Frame|Physio|Image|Face)Here>)',text)
        ids=[]
        for chunk in chunks:
            if chunk in self.special:ids.append(self.special[chunk])
            elif chunk:ids.extend([3]*max(1,len(chunk.split())))
        return {'input_ids':ids}
    def get_vocab(self):return dict(self.special)
    def add_tokens(self,tokens,special_tokens=False):
        for token in tokens:
            if token not in self.special:self.special[token]=100+len(self.special)

class EmbeddingDouble:
    def __init__(self):self.weight=Tensor(np.random.default_rng(19).normal(0,.01,(64,3584)).astype('float32'))
    def __call__(self,ids):return Tensor(self.weight[np.asarray(ids,dtype=int)])

class LlamaDouble:
    def __init__(self):self.model=SimpleNamespace(model=SimpleNamespace(embed_tokens=EmbeddingDouble()))
    def __call__(self,inputs_embeds,attention_mask,labels,**kwargs):
        # Explicit test-only causal prefix scorer. Not a transformer replacement.
        logits=np.asarray(inputs_embeds).cumsum(axis=1)[...,:16]*.02
        targets=np.asarray(labels)[:,1:];pred=logits[:,:-1]
        mask=targets!=-100
        selected=pred[mask];target=targets[mask]
        selected=selected-selected.max(axis=1,keepdims=True)
        nll=np.log(np.exp(selected).sum(axis=1))-selected[np.arange(len(target)),target]
        return SimpleNamespace(loss=Tensor(nll.mean()),logits=Tensor(logits))

class ModelDouble:
    def __init__(self):
        self.llama_tokenizer=TokenizerDouble();self.llama_model=LlamaDouble()
        for word in ['MULTI','AUDIO','FRAME','PHYSIO']:
            setattr(self,word+'_PATCH_TOKEN_ID',self.llama_tokenizer.special['<'+word.title()+'Here>'])

class AdapterDouble:
    def __init__(self):self.weight=Tensor(np.random.default_rng(42).normal(0,.001,(66,3584)).astype('float32'))
    def __call__(self,value):return Tensor(np.asarray(value)@self.weight)

class LinearDouble:
    def __init__(self,a,b,seed):self.weight=np.random.default_rng(seed).normal(0,.02,(a,b)).astype('float32')
    def __call__(self,value):return Tensor(np.asarray(value)@self.weight)

def selected_source(path,names,extra=None):
    source=path.read_text(encoding='utf-8-sig');tree=ast.parse(source)
    nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
    if {n.name for n in nodes}!=set(names):raise AssertionError((path,names))
    module=ast.Module(body=nodes,type_ignores=[])
    namespace={'torch':torch,'np':np,'EMOTIONS':EMOTIONS}
    if extra:namespace.update(extra)
    exec(compile(ast.fix_missing_locations(module),str(path),'exec'),namespace)
    return namespace

def selected_methods(path,class_name,names,extra=None):
    tree=ast.parse(path.read_text(encoding='utf-8-sig'))
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==class_name)
    methods=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in names]
    namespace={'torch':torch,'np':np}
    if extra:namespace.update(extra)
    exec(compile(ast.fix_missing_locations(ast.Module(body=methods,type_ignores=[])),str(path),'exec'),namespace)
    return namespace
