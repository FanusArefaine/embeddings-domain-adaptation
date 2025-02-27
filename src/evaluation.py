# src/evaluation.py

import numpy as np

def compute_average_precision(retrieved_indices, relevant_indices):
    """
    retrieved_indices: list of doc indices (in ranked order)
    relevant_indices: set of doc indices that are correct
    
    Returns the Average Precision for a single query.
    """
    num_relevant = len(relevant_indices)
    if num_relevant == 0:
        return 0.0

    hits = 0
    sum_precisions = 0.0
    for i, idx_ in enumerate(retrieved_indices, start=1):
        if idx_ in relevant_indices:
            hits += 1
            precision_at_i = hits / i
            sum_precisions += precision_at_i

    return sum_precisions / num_relevant


def compute_recall_at_k(retrieved_indices, relevant_indices):
    """
    Returns recall = (# relevant docs found) / (total relevant docs).
    """
    num_relevant = len(relevant_indices)
    if num_relevant == 0:
        return 0.0
    found = len(set(retrieved_indices).intersection(relevant_indices))
    return found / num_relevant


def compute_mrr(retrieved_indices, relevant_indices):
    """
    Reciprocal rank of the first relevant doc in retrieved_indices.
    If none found, returns 0.
    """
    for rank, idx_ in enumerate(retrieved_indices, start=1):
        if idx_ in relevant_indices:
            return 1.0 / rank
    return 0.0
