from neuropivot.paths import REPO, WORKDIR, OUTER_FOLD, OUTER_FOLDS, TRAIN_SUBJECTS_PUBLIC, N_TRAIN, N_TRIALS, fold_text
from pathlib import Path
import os

import numpy as np
import torch
EMOTIONS = ['neutral', 'anger', 'happiness', 'sadness', 'calmness']

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

def metrics(labels, predictions):
    matrix = np.zeros((5, 5), dtype=np.int64)
    np.add.at(matrix, (labels, predictions), 1)
    tp = np.diag(matrix).astype(np.float64)
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    precision = np.divide(tp, predicted, out=np.zeros(5), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros(5), where=support > 0)
    f1 = np.divide(2 * tp, support + predicted, out=np.zeros(5), where=support + predicted > 0)
    return {'accuracy': float(np.mean(labels == predictions)), 'balanced_accuracy': float(recall[support > 0].mean()), 'macro_f1': float(f1.mean()), 'macro_precision': float(precision.mean()), 'macro_recall': float(recall.mean()), 'confusion_matrix': matrix.tolist(), 'per_class': {name: {'support': int(support[i]), 'predicted_count': int(predicted[i]), 'precision': float(precision[i]), 'recall': float(recall[i]), 'f1': float(f1[i])} for i, name in enumerate(EMOTIONS)}}

@torch.inference_mode()
def score_candidates(ge, model, builder, row, use_residual):
    scores, counts = ([], [])
    for class_id in range(5):
        candidate = dict(row)
        candidate['label'] = class_id
        scores.append(ge.true_nll(model, builder, candidate, use_residual))
        _, targets, _, _, _ = builder.build(ge.make_batch(candidate), use_residual=use_residual)
        counts.append(int((targets[:, 1:] != -100).sum().item()))
    scores = np.asarray(scores, dtype=np.float64)
    counts = np.asarray(counts, dtype=np.int64)
    require(np.isfinite(scores).all(), 'Non-finite candidate NLL')
    require(np.all(counts > 0), 'Candidate has no supervised tokens')
    return (scores, counts)


def predict_emotion(model, builder, row):
    """Score an aligned caller-provided AV/z row with real frozen model objects."""
    from neuropivot.inference import evaluator
    scores, counts = score_candidates(evaluator, model, builder, row, True)
    index = int(np.argmin(scores))
    return {"label": index, "emotion": EMOTIONS[index], "candidate_nll": scores, "candidate_token_counts": counts}
