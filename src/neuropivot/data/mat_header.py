from pathlib import Path
import os

"""
MAT file header inspection utilities.

Provides functions to read MAT file metadata without loading full arrays.
"""
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import scipy.io

class MATHeaderError(Exception):
    """Exception raised when MAT header inspection fails."""
    pass

def read_mat_header(mat_path: Path) -> List[Tuple[str, Tuple[int, ...], str]]:
    """
    Read MAT file header without loading data.

    Args:
        mat_path: Path to MAT file

    Returns:
        List of tuples (variable_name, shape, dtype)

    Raises:
        MATHeaderError: If file cannot be read
    """
    try:
        header = scipy.io.whosmat(str(mat_path))
        return header
    except Exception as e:
        raise MATHeaderError(f'Failed to read MAT header from {mat_path}: {e}')

def find_variable_by_shape(mat_path: Path, target_shape: Tuple[int, ...], variable_hint: Optional[str]=None) -> str:
    """
    Find variable name by shape in MAT file.

    Args:
        mat_path: Path to MAT file
        target_shape: Expected shape tuple
        variable_hint: Optional hint for error messages

    Returns:
        Variable name that matches the shape

    Raises:
        MATHeaderError: If zero or multiple candidates found
    """
    header = read_mat_header(mat_path)
    candidates = [var_name for var_name, shape, _ in header if shape == target_shape]
    if len(candidates) == 0:
        hint_msg = f" (expected variable like '{variable_hint}')" if variable_hint else ''
        raise MATHeaderError(f'No variable with shape {target_shape} found in {mat_path}{hint_msg}. Available: {[(name, shape) for name, shape, _ in header]}')
    if len(candidates) > 1:
        raise MATHeaderError(f'Multiple variables with shape {target_shape} found in {mat_path}: {candidates}. Cannot determine which to use.')
    return candidates[0]

def get_mat_info(mat_path: Path) -> Dict[str, Any]:
    """
    Get structured information about MAT file contents.

    Args:
        mat_path: Path to MAT file

    Returns:
        Dictionary with variable metadata
    """
    header = read_mat_header(mat_path)
    info = {'path': str(mat_path), 'variables': {}}
    for var_name, shape, dtype in header:
        info['variables'][var_name] = {'shape': shape, 'dtype': dtype, 'ndim': len(shape), 'size': int(scipy.io._prod(shape)) if shape else 0}
    return info

def validate_eeg_data_variable(mat_path: Path, expected_shape: Tuple[int, int, int]) -> str:
    """
    Validate and return EEG data variable name.

    Args:
        mat_path: Path to MAT file
        expected_shape: Expected shape (trials, channels, samples)

    Returns:
        Variable name containing EEG data

    Raises:
        MATHeaderError: If validation fails
    """
    return find_variable_by_shape(mat_path, expected_shape, variable_hint='seg or seg1')

def validate_label_variable(mat_path: Path, expected_shape: Tuple[int, int]) -> str:
    """
    Validate and return label variable name.

    Args:
        mat_path: Path to MAT file
        expected_shape: Expected shape (emotions, trials)

    Returns:
        Variable name containing labels

    Raises:
        MATHeaderError: If validation fails
    """
    return find_variable_by_shape(mat_path, expected_shape, variable_hint='label')
