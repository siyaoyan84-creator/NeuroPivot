from pathlib import Path
import os

import torch
import torch.nn.functional as F
from neuropivot.models.eeg import Gate3EEGPivotV2, soft_kl
VALENCE_RELATIONS = ((2, 0), (2, 3), (2, 1), (0, 3), (0, 1), (4, 3), (4, 1))
AROUSAL_RELATIONS = ((1, 4), (1, 3), (2, 4), (2, 3))

class Gate3EEGPivotV3(Gate3EEGPivotV2):
    """
    Same architecture as V2.

    Difference is supervision:

    REFED:
        real continuous VA

    EAV:
        native 5-class task
        +
        weak ordinal affect constraints

    No fixed pseudo-VA distribution is given to EAV.
    """
    pass

def ordinal_dimension_loss(va_pred, semantic_labels, relations, dimension, margin=0.05):
    """
    Weak class-level ranking loss.

    For relation (higher, lower):

        mean(VA_higher)
            >
        mean(VA_lower) + margin

    Only relations whose two classes are present in the
    current batch contribute.

    No absolute VA target is imposed.
    """
    losses = []
    for higher, lower in relations:
        high_mask = semantic_labels == higher
        low_mask = semantic_labels == lower
        if not bool(high_mask.any()) or not bool(low_mask.any()):
            continue
        high_mean = va_pred[high_mask, dimension].mean()
        low_mean = va_pred[low_mask, dimension].mean()
        losses.append(F.relu(margin - (high_mean - low_mean)))
    if not losses:
        return va_pred.sum() * 0.0
    return torch.stack(losses).mean()

def eav_ordinal_loss(va_pred, labels, margin=0.05, shuffle_mapping=None):
    semantic_labels = labels
    valence = ordinal_dimension_loss(va_pred, semantic_labels, VALENCE_RELATIONS, dimension=0, margin=margin)
    arousal = ordinal_dimension_loss(va_pred, semantic_labels, AROUSAL_RELATIONS, dimension=1, margin=margin)
    total = (valence + arousal) / 2.0
    return {'total': total, 'valence': valence, 'arousal': arousal}
