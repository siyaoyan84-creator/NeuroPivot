"""CLI parsing is dependency-light; implementations import only after validation."""
import argparse
import importlib
import json
import os
import sys
from pathlib import Path

def build_parser():
    p = argparse.ArgumentParser(prog='neuropivot')
    p.add_argument('--work-dir', type=Path, default=Path('work'), help='Independent generated-output directory')
    p.add_argument('--affectgpt-root', type=Path, help='Actual modified checkout, kept external')
    p.add_argument('--model-root', dest='base_model_root', type=Path, help='Prepared base-model directories, never downloaded automatically')
    sub = p.add_subparsers(dest='command', required=True)
    schema = json.loads(Path(__file__).with_name('cli_schema.json').read_text(encoding='utf-8'))
    for name, spec in schema.items():
        c = sub.add_parser(name)
        for flags, options in spec['args']:
            options = dict(options)
            if 'type' in options:
                options['type'] = {'Path': Path, 'int': int, 'float': float, 'str': str}[options['type']]
            c.add_argument(*flags, **options)
    return p

def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    schema = json.loads(Path(__file__).with_name("cli_schema.json").read_text(encoding="utf-8"))
    if args.affectgpt_root:
        os.environ['AFFECTGPT_ROOT'] = str(args.affectgpt_root.resolve())
    if args.base_model_root:
        os.environ['AFFECTGPT_MODEL_ROOT'] = str(args.base_model_root.resolve())
    repo = Path(__file__).resolve().parents[2]
    work = args.work_dir.resolve()
    os.environ['NEUROPIVOT_WORKDIR'] = str(work)
    if hasattr(args, 'outer_fold'):
        os.environ['NEUROPIVOT_OUTER_FOLD'] = str(args.outer_fold)
    index = argv.index(args.command)
    remaining = argv[index + 1:]

    def replace_option(flag, value=None, remove=False):
        for i, word in enumerate(remaining):
            if word == flag:
                remaining[i:i + 2] = [] if remove else [flag, str(value)]
                return
            if word.startswith(flag + '='):
                remaining[i:i + 1] = [] if remove else [flag, str(value)]
                return
    if args.command in {'export-pivots', 'train-task', 'train-residual', 'predict'}:
        replace_option('--outer-fold', remove=True)
    for flag in ['--paths-config', '--cfg-path', '--checkpoint', '--output']:
        value = getattr(args, flag[2:].replace('-', '_'), None)
        if value is not None:
            replace_option(flag, Path(value).resolve())
    sys.argv = ['neuropivot ' + args.command, *remaining]
    if args.command in {'cache-av', 'train-task', 'train-residual', 'predict'}:
        from neuropivot.integrations.affectgpt.backend import initialize
        initialize()
    if hasattr(args, 'paths_config') and (not getattr(args, 'preflight_only', False)):
        from neuropivot.config import load_config
        cfg = load_config(Path(args.paths_config))
        if any(getattr(cfg,key).resolve()!=work/value for key,value in {'manifest_dir':'manifests','split_dir':'splits','audit_dir':'audits','prediction_dir':'predictions','checkpoint_dir':'checkpoints','log_dir':'logs','feature_dir':'features','cache_dir':'cache'}.items()):
            raise ValueError('--work-dir and paths-config outputs do not select the same workspace')
    implementation = importlib.import_module(schema[args.command]['module'])
    implementation.main()
if __name__ == '__main__':
    main()
