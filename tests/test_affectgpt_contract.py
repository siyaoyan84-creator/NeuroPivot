import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from affectgpt_doubles import Tensor,torch,ModelDouble,AdapterDouble,LinearDouble,selected_source,selected_methods
from neuropivot.training.contracts import bind_anchor,assert_checkpoint_anchor
from neuropivot.integrations.affectgpt.inspect import inspect,resolve_checkout

ROOT=Path(__file__).resolve().parents[1]
ORIGINAL=ROOT.parent

class ContractTests(unittest.TestCase):
    def test_anchor_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            cfg=SimpleNamespace(checkpoint_dir=Path(d)/'checkpoints')
            contract=bind_anchor(cfg,1,42)
            self.assertEqual(bind_anchor(cfg,1,42),contract)
            with self.assertRaises(ValueError):bind_anchor(cfg,2,42)
            assert_checkpoint_anchor({'anchor_contract':contract},1)
            with self.assertRaises(ValueError):assert_checkpoint_anchor({'anchor_contract':contract},2)
            with self.assertRaises(ValueError):assert_checkpoint_anchor({},1)
            from neuropivot.training.contracts import assert_pivot_provenance
            class Fields(dict):
                @property
                def files(self):return list(self)
            fields=Fields(upstream_refed_anchor_test_subject=np.asarray(1),upstream_outer_fold=np.asarray(1),upstream_seed=np.asarray(42))
            assert_pivot_provenance(fields,Path(d),1)
            with self.assertRaises(ValueError):assert_pivot_provenance(fields,Path(d),2)

    def test_real_builder_array_and_nll_parity(self):
        source=ORIGINAL/'scripts/train_gate4_affectgpt_physio_residual_v2.py'
        if not source.exists():self.skipTest('Original comparison source not present')
        old=selected_source(source,['Builder'])['Builder']
        new=selected_source(ROOT/'src/neuropivot/integrations/affectgpt/affect_plumbing.py',['Builder'])['Builder']
        old_ge=selected_source(ORIGINAL/'scripts/eval_gate4_grouped_outer01_label_fix_v1.py',['make_batch','true_nll'])
        new_ge=selected_source(ROOT/'src/neuropivot/inference/evaluator.py',['make_batch','true_nll'])
        model=ModelDouble();task=Tensor(np.zeros((1,1,3584),dtype='float32'));adapter=AdapterDouble()
        from neuropivot.integrations.affectgpt.backend import validate_model
        validate_model(model)
        model.PHYSIO_PATCH_TOKEN_ID=999
        with self.assertRaises(ValueError):validate_model(model)
        model.PHYSIO_PATCH_TOKEN_ID=103
        with contextlib.redirect_stdout(io.StringIO()):
            a=old(model,'cpu',task,adapter);b=new(model,'cpu',task,adapter)
        rng=np.random.default_rng(17)
        batch={'label':Tensor([0,2],np.int64),'z':Tensor(rng.normal(size=(2,66)),np.float32),'audio':Tensor(rng.normal(size=(2,1,3584)),np.float32),'frame':Tensor(rng.normal(size=(2,1,3584)),np.float32),'multi':Tensor(rng.normal(size=(2,1,3584)),np.float32)}
        for use in [False,True]:
            left=a.build(batch,use_residual=use);right=b.build(batch,use_residual=use)
            for x,y in zip(left,right):np.testing.assert_array_equal(x,y)
        embeds,labels,attention,multi,delta=b.build(batch,True)
        self.assertTrue(np.all(np.asarray(labels)[:,:len(b.prompt_ids)]==-100))
        multi_slot=b.prompt_ids.index(model.MULTI_PATCH_TOKEN_ID)
        task_slot=b.prompt_ids.index(model.PHYSIO_PATCH_TOKEN_ID)
        np.testing.assert_array_equal(embeds[:,multi_slot],batch['multi'][:,0]+adapter(batch['z']))
        np.testing.assert_array_equal(embeds[:,task_slot],np.zeros((2,3584),dtype='float32'))
        # Finite-difference dependence only: not a torch-autograd qualification.
        task[:]=.25
        changed=b.build(batch,True)[0]
        np.testing.assert_array_equal(changed[:,multi_slot],embeds[:,multi_slot])
        self.assertGreater(float(np.linalg.norm(changed[:,task_slot]-embeds[:,task_slot])),0)
        row={'label':0,'z':np.asarray(batch['z'][0]),'audio':np.asarray(batch['audio'][0]),'frame':np.asarray(batch['frame'][0]),'multi':np.asarray(batch['multi'][0])}
        scores=[]
        for class_id in range(5):
            row['label']=class_id
            x=old_ge['true_nll'](model,a,row,True);y=new_ge['true_nll'](model,b,row,True)
            self.assertEqual(x,y);scores.append(y)
        self.assertTrue(np.isfinite(scores).all())
        for class_id,count in enumerate([2,3,4,2,3]):
            row['label']=class_id
            targets=b.build(new_ge['make_batch'](row),True)[1]
            self.assertEqual(int((targets[:,1:]!=-100).sum()),count)

    def test_real_task_builder_parity(self):
        source=ORIGINAL/'scripts/train_gate4_affectgpt_task_adapter_grouped_outer01_label_fix_v1.py'
        if not source.exists():self.skipTest('Original comparison source not present')
        old=selected_source(source,['TaskPromptBuilder'])['TaskPromptBuilder']
        new=selected_source(ROOT/'src/neuropivot/training/task_adapter.py',['TaskPromptBuilder'])['TaskPromptBuilder']
        model=ModelDouble();task=Tensor(np.zeros((1,1,3584),dtype='float32'))
        a=old(model,'cpu',task);b=new(model,'cpu',task)
        batch={'label':Tensor([1],np.int64),**{k:Tensor(np.ones((1,1,3584),dtype='float32')) for k in ['audio','frame','multi']}}
        for x,y in zip(a.build(batch),b.build(batch)):np.testing.assert_array_equal(x,y)

    def test_actual_attention_and_chat_route(self):
        try:root=resolve_checkout()
        except FileNotFoundError as e:self.skipTest(str(e))
        methods=selected_methods(root/'my_affectgpt/models/affectgpt.py','AffectGPT',['encode_multi_attention','encode_multi_merge'])
        model=SimpleNamespace(maybe_autocast=contextlib.nullcontext,multi_fusion_type='attention',num_multi_query_token=1,multi_video_embs=LinearDouble(8,10,1),multi_audio_embs=LinearDouble(6,10,2),attention_mlp=LinearDouble(20,10,3),fc_att=LinearDouble(10,2,4),multi_llama_proj=LinearDouble(10,3584,5))
        model.encode_multi_attention=lambda v,a:methods['encode_multi_attention'](model,v,a)
        model.encode_multi_merge=lambda v,a:methods['encode_multi_merge'](model,v,a)
        rng=np.random.default_rng(3);v=Tensor(rng.normal(size=(2,8,8)),np.float32);a=Tensor(rng.normal(size=(2,12,6)),np.float32)
        chat_method=selected_methods(root/'my_affectgpt/conversation/conversation_video.py','Chat',['postprocess_multi'])['postprocess_multi']
        hidden,output=chat_method(SimpleNamespace(model=model),v,a)
        video=model.multi_video_embs(np.asarray(v).mean(axis=1));audio=model.multi_audio_embs(np.asarray(a).mean(axis=1))
        raw_attention=model.fc_att(model.attention_mlp(np.concatenate([video,audio],axis=1)))
        expected=(np.stack([video,audio],axis=2)@raw_attention[:,:,None])[:,:,0]
        np.testing.assert_array_equal(hidden,expected)
        np.testing.assert_array_equal(output,model.multi_llama_proj(expected)[:,None,:])
        self.assertEqual(output.shape,(2,1,3584))

    def test_one_and_five_fold_cli_chain(self):
        import subprocess,json,sys
        from neuropivot.cli import build_parser
        parser=build_parser()
        checked=subprocess.run([sys.executable,str(ROOT/'scripts/neuropivot.py'),'bridge','--outer-fold','1','--split-role','outer','--preflight-only','--paths-config',str(ROOT/'configs/paths.example.yaml'),'--mode','ordinal_correct','--refed-anchor-test-subject','1'],capture_output=True,text=True)
        self.assertEqual(checked.returncode,0,checked.stderr)
        for folds in [[1],[1,2,3,4,5]]:
            result=subprocess.run([sys.executable,str(ROOT/'scripts/run_pipeline.py'),'--anchor','1','--folds',*map(str,folds),'--paths-config',str(ROOT/'configs/paths.example.yaml'),'--base-checkpoint','not-loaded-in-plan.pth'],capture_output=True,text=True,check=True)
            plan=json.loads(result.stdout)
            self.assertFalse(plan['executed'])
            bridge=[];pivots=[]
            for stage in plan['stages']:
                argv=stage['argv']
                if argv[1].endswith('neuropivot.py'):
                    parsed=parser.parse_args(argv[2:])
                    if parsed.command=='bridge':bridge.append(parsed)
                    if parsed.command=='export-pivots':pivots.append(parsed)
            self.assertEqual(len(bridge),6*len(folds));self.assertEqual(len(pivots),len(folds))
            self.assertTrue(all(x.refed_anchor_test_subject==1 for x in bridge+pivots))

    def test_reviewed_import_and_token_contract(self):
        try:report=inspect()
        except FileNotFoundError as e:self.skipTest(str(e))
        self.assertFalse(report['missing_required']);self.assertFalse(report['missing_local_imports'])
        self.assertEqual(report['special_tokens']['DEFAULT_PHYSIO_PATCH_TOKEN'],'<PhysioHere>')
        self.assertEqual(report['model_arch'],'affectgpt')
        self.assertEqual(report['model_assets']['Qwen2.5-7B-Instruct'].get('hidden_size'),3584)

if __name__=='__main__':unittest.main()
