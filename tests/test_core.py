import tempfile
import unittest
from pathlib import Path
import numpy as np
from neuropivot.data.normalization import Normalization
from neuropivot.training import protocol

class CoreTests(unittest.TestCase):
    def test_frozen_grouped_crossfit(self):
        splits=protocol.validate_protocol()
        outer=set()
        for fold in range(1,6):
            test=set(protocol.OUTER_FOLDS[fold])
            self.assertFalse(test & outer)
            outer |= test
            train=set(range(3,43))-test
            held=set()
            for role in ['inner1','inner2','inner3','inner4','inner5']:
                split=protocol.build_split(fold,role)
                a,b,c=map(set,(split['train'],split['val'],split['test']))
                self.assertEqual(b,{2})
                self.assertFalse(a&b or a&c or b&c or test&(a|b|c))
                self.assertFalse(held&c)
                held |= c
                self.assertEqual(a|c,train)
            self.assertEqual(held,train)
        self.assertEqual(outer,set(range(4,43)))

    def test_normalization_roundtrip_and_leakage(self):
        rng=np.random.default_rng(42)
        for kind,x in [('eeg',rng.normal(size=(8,30,500)).astype('float32')),('pivot',rng.normal(size=(8,66)).astype('float32'))]:
            n=Normalization.fit(x,kind,[3]*8,[3,4])
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'stats.npz';n.save(path)
                restored=Normalization.load(path)
                np.testing.assert_array_equal(n.transform(x),restored.transform(x))
            with self.assertRaises(ValueError):
                Normalization.fit(x,kind,[2]*8,[3,4])

    def test_cache_specs_match_grouped_protocol(self):
        import json
        specs=json.loads((Path(__file__).resolve().parents[1]/'configs/folds.json').read_text())
        for fold in range(1,6):
            for role,subjects in specs[str(fold)]['ROLE_TESTS'].items():
                split=protocol.build_split(fold,role)
                self.assertEqual(split['test'],subjects)
                self.assertEqual(protocol.job_id_for(fold,role),specs[str(fold)]['ROLE_JOB_IDS'][role])

    def test_trial_aggregation_and_alignment(self):
        try:
            import torch
        except ImportError as exc:
            self.skipTest('PyTorch unavailable: '+str(exc))
        from neuropivot.inference.pivot_cache_base import aggregate_trials
        from neuropivot.inference.export_pivots import aligned_metadata
        cache={'sequence_id':np.repeat(np.arange(101,201),4),'label':np.repeat(np.arange(100)%5,4),'window_index':np.tile(np.arange(4),100),'parent_trial_group_id':np.repeat(np.asarray(['trial'+str(i) for i in range(100)]),4)}
        va=np.arange(800,dtype='float32').reshape(400,2)
        latent=np.arange(25600,dtype='float32').reshape(400,64)
        trial=aggregate_trials(cache,va,latent)
        np.testing.assert_array_equal(trial['teacher_va'],va.reshape(100,4,2).mean(axis=1))
        np.testing.assert_array_equal(trial['pivot_shared'],latent.reshape(100,4,64).mean(axis=1))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'metadata.npz'
            payload={k:trial[k] for k in ['sequence_id','parent_trial_group_id','label']}
            payload.update(audio_path=np.asarray(['audio']*100),video_path=np.asarray(['video']*100))
            np.savez(path,**payload)
            aligned_metadata(path,trial,3)
            payload['sequence_id']=payload['sequence_id'][::-1]
            np.savez(path,**payload)
            with self.assertRaises(RuntimeError):
                aligned_metadata(path,trial,3)
        cache['window_index'][0]=3
        with self.assertRaises(RuntimeError):
            aggregate_trials(cache,va,latent)

    def test_checkpoint_and_real_network_shapes(self):
        try:
            import torch
        except ImportError as exc:
            self.skipTest('PyTorch unavailable: '+str(exc))
        from neuropivot.models.ordinal import Gate3EEGPivotV3
        from neuropivot.models.residual import ResidualAdapter
        from neuropivot.inference.interfaces import EEGEncoder,AVConditioner
        torch.manual_seed(42)
        model=Gate3EEGPivotV3().eval()
        mean=np.zeros((1,30,1),dtype='float32');std=np.ones_like(mean)
        encoder=EEGEncoder(model,mean,std)
        x=torch.randn(2,30,500)
        out=encoder.encode_eeg(x)
        self.assertEqual(tuple(out['va'].shape),(2,2))
        self.assertEqual(tuple(out['latent'].shape),(2,64))
        self.assertEqual(tuple(out['pivot'].shape),(2,66))
        self.assertTrue(torch.equal(out['pivot'],torch.cat([out['va'],out['latent']],dim=1)))
        self.assertTrue(bool(((out['va']>=0)&(out['va']<=1)).all()))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'eeg.pt'
            torch.save({'model_state':model.state_dict(),'public_format_version':1,'normalization':{'mean':torch.from_numpy(mean),'std':torch.from_numpy(std),'dataset':'eav'}},path)
            restored=EEGEncoder.from_checkpoint(path)
            self.assertTrue(torch.equal(out['pivot'],restored.encode_eeg(x)['pivot']))
            adapter=ResidualAdapter(66)
            conditioner=AVConditioner(adapter,np.zeros(66),np.ones(66))
            av=torch.randn(2,1,3584)
            self.assertTrue(torch.equal(conditioner.condition_av(av,out),av))
            with torch.no_grad():
                adapter.fc2.weight.normal_(0,0.01)
            with torch.no_grad():
                expected=av+adapter(out['pivot']).unsqueeze(1)
            self.assertTrue(torch.allclose(conditioner.condition_av(av,out),expected))
            path=Path(tmp)/'adapter.pt'
            torch.save({'adapter':adapter.state_dict(),'feature_dim':66,'representation':'va_pivot','mode':'real','feature_mean':torch.zeros(66),'feature_std':torch.ones(66)},path)
            restored=AVConditioner.from_checkpoint(path)
            self.assertTrue(torch.equal(restored.condition_av(av,out),conditioner.condition_av(av,out)))
        self.assertEqual(tuple(EEGEncoder(model,np.zeros((1,64,1)),np.ones((1,64,1)),'refed').encode_eeg(torch.randn(2,64,200))['pivot'].shape),(2,66))

if __name__=='__main__':
    unittest.main()
