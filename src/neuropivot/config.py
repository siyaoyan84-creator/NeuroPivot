from neuropivot.paths import REPO, WORKDIR, OUTER_FOLD, OUTER_FOLDS, TRAIN_SUBJECTS_PUBLIC, N_TRAIN, N_TRIALS, fold_text
from pathlib import Path
import os

"""
Central configuration loader for neuro-affect-pivot.

Loads paths and other configurations from YAML files.
Supports both local and server deployment.
"""
from pathlib import Path
from typing import Dict, Any

class PathConfig:
    """Path configuration with validation."""

    def __init__(self, config_dict: Dict[str, Any]):
        """
        Initialize path configuration from dictionary.

        Args:
            config_dict: Dictionary containing path configurations

        Raises:
            KeyError: If required configuration keys are missing
        """
        self._config = config_dict
        if 'data' not in config_dict:
            raise KeyError("Missing required 'data' section in configuration")
        data_config = config_dict['data']
        if 'eav_root' not in data_config:
            raise KeyError("Missing required 'data.eav_root' in configuration")
        if 'refed_root' not in data_config:
            raise KeyError("Missing required 'data.refed_root' in configuration")
        self.eav_root = Path(data_config['eav_root'])
        self.refed_root = Path(data_config['refed_root'])
        if 'outputs' not in config_dict:
            raise KeyError("Missing required 'outputs' section in configuration")
        outputs_config = config_dict['outputs']
        required_outputs = ['manifest_dir', 'split_dir', 'audit_dir', 'prediction_dir', 'checkpoint_dir', 'log_dir', 'feature_dir', 'cache_dir']
        for key in required_outputs:
            if key not in outputs_config:
                raise KeyError(f"Missing required 'outputs.{key}' in configuration")
        self.manifest_dir = Path(outputs_config['manifest_dir'])
        self.split_dir = Path(outputs_config['split_dir'])
        self.audit_dir = Path(outputs_config['audit_dir'])
        self.prediction_dir = Path(outputs_config['prediction_dir'])
        self.checkpoint_dir = Path(outputs_config['checkpoint_dir'])
        self.log_dir = Path(outputs_config['log_dir'])
        self.feature_dir = Path(outputs_config['feature_dir'])
        self.cache_dir = Path(outputs_config['cache_dir'])

    def ensure_output_dirs(self) -> None:
        """Create output directories if they don't exist."""
        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        self.split_dir.mkdir(parents=True, exist_ok=True)
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        self.prediction_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.feature_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        (self.audit_dir / 'gate0').mkdir(parents=True, exist_ok=True)

    def validate_input_paths(self, dataset: str) -> None:
        """
        Validate that required input paths exist and are readable.

        Args:
            dataset: "eav" or "refed"

        Raises:
            FileNotFoundError: If required paths don't exist
        """
        if dataset == 'eav':
            if not self.eav_root.exists():
                raise FileNotFoundError(f'EAV root not found: {self.eav_root}')
        elif dataset == 'refed':
            if not self.refed_root.exists():
                raise FileNotFoundError(f'REFED root not found: {self.refed_root}')
        else:
            raise ValueError(f'Unknown dataset: {dataset}')

def load_config(config_path: Path) -> PathConfig:
    import yaml
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        PathConfig instance

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid
        KeyError: If required configuration keys are missing
    """
    if not config_path.exists():
        raise FileNotFoundError(f'Config file not found: {config_path}')
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    return PathConfig(config_dict)
