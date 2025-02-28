import numpy as np

from src.config import Config
from src.data import (
    load_testing_data,
    get_questions_and_context_docs
)
from src.finetuning import (
    mnr_loss_finetuning,
    cosine_loss_finetuning,
    softmax_loss_finetuning,
    # if you have more
)
from src.embedding import embed_texts
from src.retrieval import build_faiss_index, search_top_k
from src.evaluation import compute_average_precision, compute_recall_at_k, compute_mrr

def run_finetuning_experiment(cfg: Config, loss_type: str = "mnr"):
    """
    A generic pipeline that:
      1) Fine-tunes a SentenceTransformer using the chosen loss type
      2) Loads the test set
      3) Embeds & indexes test docs
      4) Evaluates retrieval (mAP, Recall@k, MRR)
    
    Args:
        cfg (Config): experiment config
        loss_type (str): one of ["mnr", "cosine", "softmax"] (or others you define)
    """

    # 1) Fine-tune the model
    if loss_type == "mnr":
        finetuned_model = mnr_loss_finetuning(cfg)
        loss_name = "MNR"
    elif loss_type == "cosine":
        finetuned_model = cosine_loss_finetuning(cfg)
        loss_name = "CosineSimilarity"
    elif loss_type == "softmax":
        finetuned_model = softmax_loss_finetuning(cfg)
        loss_name = "Softmax"
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")

    print(f"Finished fine-tuning with {loss_name} loss.")

    # 2) Load test data
    test_df = load_testing_data(cfg)
    print(f"Loaded test data: {len(test_df)} rows")

    # 3) Create doc & question objects for the test set
    test_contexts, test_questions = get_questions_and_context_docs(cfg, test_df)
    print(f"Test questions: {len(test_questions)}")
    print(f"Test context docs: {len(test_contexts)}")

    # Embed doc texts with the newly finetuned model
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

        q_emb = embed_texts(finetuned_model, [question_text], do_normalize=cfg.NORMALIZE)
        idxs, _ = search_top_k(gpu_index, q_emb, top_k=cfg.TOP_K, do_normalize=False)
        retrieved_indices = idxs[0].tolist()

        # relevant docs = all docs with the same pubid
        relevant_indices = {i for i, pid in enumerate(pubid_map) if pid == question_pubid}

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

    print(f"\nEvaluation on test set with {loss_name} fine-tuned model:")
    print(f"mAP@{cfg.TOP_K}: {mAP:.4f}")
    print(f"Recall@{cfg.TOP_K}: {mean_recall:.4f}")
    print(f"MRR@{cfg.TOP_K}: {mean_mrr:.4f}")
