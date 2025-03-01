# src/pipeline.py

import numpy as np
from src.config import Config
from src.data import load_testing_data, get_questions_and_context_docs
from src.embedding import build_embedder, embed_texts
from src.retrieval import build_faiss_index, search_top_k
from src.evaluation import compute_average_precision, compute_recall_at_k, compute_mrr

def run_experiment(cfg: Config):
    """
    Main pipeline for the base experimentation:
    1) Load & filter data
    2) Create docs & questions
    3) Build embedder, embed doc texts
    4) Build FAISS index
    5) For each question, retrieve top-k, compute metrics
    6) Print final average metrics
    """
    # 1) Load testing data split
    df = load_testing_data(cfg)
    print(f"Testing data size: {len(df)}")

    # Rest of the code remains the same
    # 2) Docs & questions
    all_context_docs, questions_data = get_questions_and_context_docs(cfg, df)
    print(f"Number of questions: {len(questions_data)}")
    print(f"Number of context docs: {len(all_context_docs)}")

    # 3) Embed docs
    embedder = build_embedder(cfg)
    doc_texts = [doc['text'] for doc in all_context_docs]
    doc_embeddings = embed_texts(embedder, doc_texts, do_normalize=cfg.NORMALIZE)

    # 4) Build index
    gpu_index = build_faiss_index(doc_embeddings)
    print(f"FAISS index size: {gpu_index.ntotal}")

    # Mappings for quick reference
    pubid_map = [doc['pubid'] for doc in all_context_docs]

    # 5) Retrieve for each question, compute metrics
    all_ap, all_recall, all_mrr = [], [], []

    for q_data in questions_data:
        question_text = q_data['question']
        question_pubid = q_data['pubid']

        q_emb = embed_texts(embedder, [question_text], do_normalize=cfg.NORMALIZE)
        idxs, _ = search_top_k(gpu_index, q_emb, top_k=cfg.TOP_K, do_normalize=False)
        # idxs shape is (1, top_k)
        retrieved_indices = idxs[0].tolist()

        # relevant docs = all doc indices with same pubid
        relevant_indices = {i for i, pid in enumerate(pubid_map) if pid == question_pubid}

        ap = compute_average_precision(retrieved_indices, relevant_indices)
        recall_k = compute_recall_at_k(retrieved_indices, relevant_indices)
        mrr = compute_mrr(retrieved_indices, relevant_indices)

        all_ap.append(ap)
        all_recall.append(recall_k)
        all_mrr.append(mrr)

    # 6) Aggregate
    mAP = np.mean(all_ap)
    mean_recall = np.mean(all_recall)
    mean_mrr = np.mean(all_mrr)

    print(f"\nResults with top_k={cfg.TOP_K}:")
    print(f"mAP: {mAP:.4f}")
    print(f"Recall@{cfg.TOP_K}: {mean_recall:.4f}")
    print(f"MRR: {mean_mrr:.4f}")