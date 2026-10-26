from neuropivot.paths import REPO, WORKDIR, OUTER_FOLD, OUTER_FOLDS, TRAIN_SUBJECTS_PUBLIC, N_TRAIN, N_TRIALS, fold_text
from pathlib import Path
import os

"""
Script to build trial-level manifests for EAV and REFED datasets.

Usage:
    python scripts/build_manifest.py --dataset eav --paths-config configs/paths.example.yaml
    python scripts/build_manifest.py --dataset refed --paths-config configs/paths.example.yaml
"""
import argparse
import csv
import json
import sys
from pathlib import Path
from neuropivot.config import load_config
from neuropivot.data import eav_manifest, refed_manifest
from neuropivot.data.schema import EAVTrial, REFEDTrial

def write_csv_and_summary_atomic(trials, manifest_path, summary_path, field_names, summary):
    """
    Write CSV and summary JSON atomically.

    Both files are written to temp locations first, then both are replaced atomically.
    If any write fails, neither file is modified.

    Args:
        trials: List of trial records
        manifest_path: Final CSV path
        summary_path: Final JSON path
        field_names: CSV field names
        summary: Summary dictionary

    Raises:
        Exception: If writing fails, temp files are cleaned up
    """
    csv_tmp = manifest_path.with_suffix('.csv.tmp')
    json_tmp = summary_path.with_suffix('.json.tmp')
    try:
        with open(csv_tmp, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=field_names)
            writer.writeheader()
            for trial in trials:
                writer.writerow(trial.to_dict())
        with open(json_tmp, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        csv_tmp.replace(manifest_path)
        json_tmp.replace(summary_path)
    except Exception:
        if csv_tmp.exists():
            csv_tmp.unlink()
        if json_tmp.exists():
            json_tmp.unlink()
        raise

def main():
    parser = argparse.ArgumentParser(description='Build trial-level manifest for EAV or REFED dataset')
    parser.add_argument('--dataset', required=True, choices=['eav', 'refed'], help='Dataset to build manifest for')
    parser.add_argument('--paths-config', required=True, type=Path, help='Path to YAML configuration file')
    args = parser.parse_args()
    try:
        config = load_config(args.paths_config)
    except Exception as e:
        print(f'Failed to load configuration: {e}', file=sys.stderr)
        return 1
    config.ensure_output_dirs()
    try:
        config.validate_input_paths(args.dataset)
    except FileNotFoundError as e:
        print(f'Input validation failed: {e}', file=sys.stderr)
        return 1
    try:
        if args.dataset == 'eav':
            print('Building EAV manifest...')
            trials = eav_manifest.build_eav_manifest(config.eav_root)
            print('Validating EAV dataset completeness...')
            eav_manifest.validate_eav_manifest_dataset(trials)
            print('Generating summary...')
            summary = eav_manifest.generate_eav_summary(trials)
            field_names = EAVTrial.field_names()
            manifest_path = config.manifest_dir / 'eav_manifest.csv'
            summary_path = config.audit_dir / 'gate0' / 'eav_manifest_summary.json'
        else:
            print('Building REFED manifest...')
            trials = refed_manifest.build_refed_manifest(config.refed_root)
            print('Validating REFED dataset completeness...')
            refed_manifest.validate_refed_manifest_dataset(trials)
            print('Generating summary...')
            summary = refed_manifest.generate_refed_summary(trials)
            field_names = REFEDTrial.field_names()
            manifest_path = config.manifest_dir / 'refed_manifest.csv'
            summary_path = config.audit_dir / 'gate0' / 'refed_manifest_summary.json'
        print('Writing manifest and summary...')
        write_csv_and_summary_atomic(trials, manifest_path, summary_path, field_names, summary)
        print(f'\nManifest saved to: {manifest_path}')
        print(f'Summary saved to: {summary_path}')
        with open(manifest_path, 'r', encoding='utf-8') as f:
            line_count = sum((1 for _ in f))
        print(f'Total lines (including header): {line_count}')
        print('\n=== Manifest Summary ===')
        for key, value in summary.items():
            print(f'{key}: {value}')
        print('\n✓ Manifest generation completed successfully!')
        return 0
    except (eav_manifest.EAVManifestError, refed_manifest.REFEDManifestError) as e:
        print(f'\nManifest generation failed: {e}', file=sys.stderr)
        print('CSV and summary files were not written.', file=sys.stderr)
        return 1
    except Exception as e:
        print(f'\nUnexpected error: {e}', file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1
if __name__ == '__main__':
    sys.exit(main())
