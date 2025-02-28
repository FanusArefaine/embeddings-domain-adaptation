# src/mnr_pipeline.py

import pandas as pd
import numpy as np
import os

from src.config import Config
from src.data import load_and_filter_data, create_documents_and_questions, train_test_split_pubmedqa, build_mnr_samples
from src.fine_tuning import fine_tune_mnr
from src.embedding import embed_texts, build_embedder
from src.retrieval import build_faiss_index, search_top_k
from src.evaluation import compute_average_precision, compute_recall_at_k, compute_mrr

def run_mnr_experiment(cfg: Config):
    """
    1) Load entire data (14k), split into train(12k) & test(2k).
    2) Build MNR train samples from train set.
    3) Fine-tune model with MNR.
    4) Evaluate the fine-tuned model on the same 2k test set used in baseline.
    """
    # 1) Load entire data
    full_df = load_and_filter_data(cfg)  # if this loads all 14k
    print(f"Total loaded: {len(full_df)}")

    # 2) Split
    train_df, test_df = train_test_split_pubmedqa(full_df, test_size=2000)
    print(f"Train size: {len(train_df)} | Test size: {len(test_df)}")

    # Build MNR samples from train
    train_samples = build_mnr_samples(train_df)
    print(f"MNR training samples: {len(train_samples)}")

    # 3) Fine-tune
    finetuned_model = fine_tune_mnr(cfg, train_samples)
    print("Finished fine-tuning. Model saved at:", cfg.MODEL_OUTPUT_DIR)

    # 4) Evaluate on the test set
    # Create question-doc pairs for test
    test_contexts, test_questions = create_documents_and_questions(test_df)

    # Build embeddings with the newly finetuned model
    doc_texts = [doc['text'] for doc in test_contexts]
    doc_embeddings = embed_texts(finetuned_model, doc_texts, do_normalize=cfg.NORMALIZE)

    gpu_index = build_faiss_index(doc_embeddings)
    pubid_map = [doc['pubid'] for doc in test_contexts]

    all_ap, all_recall, all_mrr = [], [], []
    for q_data in test_questions:
        q_text = q_data['question']
        q_emb = embed_texts(finetuned_model, [q_text], do_normalize=cfg.NORMALIZE)
        idxs, _ = search_top_k(gpu_index, q_emb, cfg.TOP_K, do_normalize=False)
        retrieved_indices = idxs[0].tolist()
        
        # relevant = all doc indices that share pubid
        relevant_indices = {i for i, pid in enumerate(pubid_map) if pid == q_data['pubid']}
        
        ap = compute_average_precision(retrieved_indices, relevant_indices)
        rec = compute_recall_at_k(retrieved_indices, relevant_indices)
        mrr = compute_mrr(retrieved_indices, relevant_indices)
        
        all_ap.append(ap)
        all_recall.append(rec)
        all_mrr.append(mrr)

    # Summarize
    mAP = np.mean(all_ap)
    mean_recall = np.mean(all_recall)
    mean_mrr = np.mean(all_mrr)
    print("\nFine-tuned MNR model on test set:")
    print(f"mAP@{cfg.TOP_K}: {mAP:.4f}")
    print(f"Recall@{cfg.TOP_K}: {mean_recall:.4f}")
    print(f"MRR@{cfg.TOP_K}: {mean_mrr:.4f}")
