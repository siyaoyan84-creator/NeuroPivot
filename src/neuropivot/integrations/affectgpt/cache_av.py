from neuropivot.integrations.affectgpt.backend import validate_model
from neuropivot.paths import REPO, WORKDIR, OUTER_FOLD, OUTER_FOLDS, TRAIN_SUBJECTS_PUBLIC, N_TRAIN, N_TRIALS, fold_text
from pathlib import Path
import os
from neuropivot.integrations.affectgpt.backend import Config, registry
from neuropivot.integrations.affectgpt import backend
import os
import sys
import argparse
from pathlib import Path
import numpy as np
import torch
ROOT = REPO
AFFECTGPT_ROOT = Path(os.environ.get('AFFECTGPT_ROOT', 'third_party/AffectGPT/AffectGPT'))
FACE_OR_FRAME = 'multiframe_audio_frame_text'

def parse_subjects(text):
    out = []
    for part in text.split(','):
        part = part.strip()
        if '-' in part:
            a, b = map(int, part.split('-', 1))
            out.extend(range(a, b + 1))
        else:
            out.append(int(part))
    return sorted(set(out))

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--cfg-path', default='train_configs/mercaptionplus_outputhybird_bestsetup_bestfusion_frame_lz.yaml')
    p.add_argument('--options', nargs='+', default=['inference.test_epoch=30', 'inference.gpu=0'])
    p.add_argument('--checkpoint', default='output/mercaptionplus_outputhybird_bestsetup_bestfusion_frame_lz/mercaptionplus_outputhybird_bestsetup_bestfusion_frame_lz_20250408110/checkpoint_000030_loss_0.751.pth')
    p.add_argument('--subjects', default='2-42')
    p.add_argument('--max-trials', type=int, default=None)
    p.add_argument('--overwrite', action='store_true')
    return p.parse_args()

def make_dataset_helper(cfg):
    ds = backend.MER2025OV_Dataset.__new__(backend.MER2025OV_Dataset)
    ds.needed_data = ds.get_needed_data(FACE_OR_FRAME)
    ds.vis_processor = backend.BaseProcessor()
    ds.img_processor = backend.BaseProcessor()
    vis_cfg = cfg.inference_cfg.get('vis_processor')
    img_cfg = cfg.inference_cfg.get('img_processor')
    if vis_cfg is not None:
        ds.vis_processor = registry.get_processor_class(vis_cfg.train.name).from_config(vis_cfg.train)
    if img_cfg is not None:
        ds.img_processor = registry.get_processor_class(img_cfg.train.name).from_config(img_cfg.train)
    ds.n_frms = cfg.model_cfg.vis_processor.train.n_frms
    return ds

def main():
    args = parse_args()
    assert torch.cuda.is_available()
    assert torch.cuda.device_count() == 1
    subjects = parse_subjects(args.subjects)
    if 1 in subjects:
        raise RuntimeError('subject01 is reserved; do not cache it.')
    device = 'cuda:0'
    print('=' * 80)
    print('GATE4 AFFECTGPT FROZEN A/V CACHE')
    print('=' * 80)
    print('GPU:', torch.cuda.get_device_name(0))
    print('subjects:', subjects)
    cfg = Config(args)
    model_cfg = cfg.model_cfg
    ckpt = Path(args.checkpoint)
    assert ckpt.is_file(), ckpt
    model_cfg.ckpt_3 = str(ckpt)
    model_cls = registry.get_model_class(model_cfg.arch)
    model = model_cls.from_config(model_cfg)
    validate_model(model)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    chat = backend.Chat(model, model_cfg, device=device)
    ds = make_dataset_helper(cfg)
    teacher_root = WORKDIR / 'features' / 'gate4' / 'eav_trial_metadata_v1'
    out_root = WORKDIR / 'features' / 'gate4' / 'gate4_affectgpt_av_llm_embeddings_v1'
    out_root.mkdir(parents=True, exist_ok=True)
    for subject in subjects:
        inp = teacher_root / f'subject{subject:02d}.npz'
        assert inp.is_file(), inp
        out = out_root / f'subject{subject:02d}.npz'
        if out.exists() and (not args.overwrite):
            print(f'subject{subject:02d}: already exists, skip')
            continue
        z = np.load(inp, allow_pickle=True)
        seq = z['sequence_id'].astype(np.int64)
        label = z['label'].astype(np.int64)
        audio_paths = z['audio_path'].astype(str)
        video_paths = z['video_path'].astype(str)
        assert len(seq) == 100
        n = len(seq)
        if args.max_trials is not None:
            n = min(n, args.max_trials)
        audio_all = []
        frame_all = []
        multi_all = []
        print()
        print('=' * 80)
        print(f'SUBJECT {subject:02d}: {n} trials')
        print('=' * 80)
        for i in range(n):
            video = Path(video_paths[i])
            audio = Path(audio_paths[i])
            assert video.is_file(), video
            assert audio.is_file(), audio
            sample_data = ds.read_frame_face_audio_text(video_path=str(video), face_npy=None, audio_path=str(audio), image_path=None)
            with torch.inference_mode():
                audio_hiddens, audio_llms = chat.postprocess_audio(sample_data)
                frame_hiddens, frame_llms = chat.postprocess_frame(sample_data)
                _, multi_llms = chat.postprocess_multi(frame_hiddens, audio_hiddens)
            assert audio_llms.shape[0] == 1
            assert frame_llms.shape[0] == 1
            assert multi_llms.shape[0] == 1
            audio_all.append(audio_llms[0].detach().cpu().half().numpy())
            frame_all.append(frame_llms[0].detach().cpu().half().numpy())
            multi_all.append(multi_llms[0].detach().cpu().half().numpy())
            if i == 0 or (i + 1) % 10 == 0 or i + 1 == n:
                print(f'subject{subject:02d} {i + 1:03d}/{n:03d}')
        audio_all = np.stack(audio_all)
        frame_all = np.stack(frame_all)
        multi_all = np.stack(multi_all)
        if audio_all.shape!=(n,1,3584) or frame_all.shape!=(n,1,3584) or multi_all.shape!=(n,1,3584):
            raise ValueError('AffectGPT AV export must produce [N,1,3584] for every slot')
        assert np.isfinite(audio_all).all()
        assert np.isfinite(frame_all).all()
        assert np.isfinite(multi_all).all()
        np.savez_compressed(out, subject_number=np.full(n, subject, dtype=np.int64), sequence_id=seq[:n], label=label[:n], audio_llms=audio_all, frame_llms=frame_all, multi_llms=multi_all, audio_path=audio_paths[:n], video_path=video_paths[:n])
        print('saved:', out)
        print('audio:', audio_all.shape, audio_all.dtype)
        print('frame:', frame_all.shape, frame_all.dtype)
        print('multi:', multi_all.shape, multi_all.dtype)
    print()
    print('=' * 80)
    print('GATE4 FROZEN A/V CACHE: PASS')
    print('=' * 80)
if __name__ == '__main__':
    main()
