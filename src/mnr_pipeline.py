# src/mnr_pipeline.py

import numpy as np

from src.config import Config
from src.data import (
    load_and_filter_data,
    train_test_split_pubmedqa,
    get_questions_and_context_docs,
    load_training_data,
    load_testing_data
)
from src.finetuning import mnr_loss_finetuning
from src.embedding import embed_texts
from src.retrieval import build_faiss_index, search_top_k
from src.evaluation import compute_average_precision, compute_recall_at_k, compute_mrr

def run_mnr_experiment(cfg: Config):
    """
    Runs an MNR-based pipeline analogous to the base pipeline:
     1) Load entire PubMedQA (filtered to yes/no, 3 contexts)
     2) Split into train & test
     3) Build MNR samples & fine-tune model
     4) Evaluate retrieval on test set with the new fine-tuned model.
    """

    # fine-tune the model (returns a SentenceTransformer)
    finetuned_model = mnr_loss_finetuning(cfg)
    print("Finished fine-tuning MNR model.")

    # 4) Evaluate on the test set
    #   - Use the same approach as the base pipeline:
    #   - Create doc & question objects, embed docs, build FAISS index, measure retrieval metrics.
    
    # Load test data
    test_df = load_testing_data(cfg)
    print(f"Loaded test data: {len(test_df)}")

    # Create test contexts & questions
    test_contexts, test_questions = get_questions_and_context_docs(cfg, test_df)
    print(f"Test questions: {len(test_questions)}")
    print(f"Test context docs: {len(test_contexts)}")

    # Embed doc texts with the newly fine-tuned model
    doc_texts = [doc['text'] for doc in test_contexts]
    doc_embeddings = embed_texts(finetuned_model, doc_texts, do_normalize=cfg.NORMALIZE)

    # Build FAISS index
    gpu_index = build_faiss_index(doc_embeddings)
    print(f"FAISS index size: {gpu_index.ntotal}")

    # Evaluate retrieval
    pubid_map = [doc["pubid"] for doc in test_contexts]
    all_ap, all_recall, all_mrr = [], [], []

    for q_data in test_questions:
        question_text = q_data['question']
        question_pubid = q_data['pubid']

        # Embed the question
        q_emb = embed_texts(finetuned_model, [question_text], do_normalize=cfg.NORMALIZE)
        idxs, _ = search_top_k(gpu_index, q_emb, top_k=cfg.TOP_K, do_normalize=False)

        retrieved_indices = idxs[0].tolist()
        # Relevant indices = all docs that share the same pubid
        relevant_indices = {i for i, pid in enumerate(pubid_map) if pid == question_pubid}

        # Compute metrics
        ap = compute_average_precision(retrieved_indices, relevant_indices)
        recall_k = compute_recall_at_k(retrieved_indices, relevant_indices)
        mrr_val = compute_mrr(retrieved_indices, relevant_indices)

        all_ap.append(ap)
        all_recall.append(recall_k)
        all_mrr.append(mrr_val)

    # Summarize
    mAP = np.mean(all_ap)
    mean_recall = np.mean(all_recall)
    mean_mrr = np.mean(all_mrr)

    print("\nEvaluation on test set with fine-tuned MNR model:")
    print(f"mAP@{cfg.TOP_K}: {mAP:.4f}")
    print(f"Recall@{cfg.TOP_K}: {mean_recall:.4f}")
    print(f"MRR@{cfg.TOP_K}: {mean_mrr:.4f}")
