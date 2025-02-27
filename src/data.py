# src/data.py

import pandas as pd
from datasets import load_dataset

def load_and_filter_data(cfg):
    """
    1. Loads the PubMedQA dataset from Hugging Face.
    2. Converts to a pandas DataFrame.
    3. Filters rows by final_decision, context count.
    4. Samples if dataset is larger than cfg.SAMPLE_SIZE.

    Returns:
        pd.DataFrame
    """
    # 1) Load
    pubmedqa = load_dataset(cfg.DATASET_NAME, cfg.DATASET_CONFIG)
    train_dataset = pubmedqa[cfg.SPLIT_NAME]

    # 2) Convert to DataFrame
    df = train_dataset.to_pandas()

    # 3) Filter
    df = df[df["final_decision"].isin(cfg.DECISIONS)]
    df["num_contexts"] = df["context"].apply(lambda x: len(x["contexts"]))
    df = df[df["num_contexts"] == cfg.CONTEXT_COUNT].copy()
    df.drop_duplicates(subset="pubid", keep="first", inplace=True)

    # 4) (Optional) sample
    if cfg.SAMPLE_SIZE is not None and len(df) > cfg.SAMPLE_SIZE:
        df = df.sample(n=cfg.SAMPLE_SIZE, random_state=42).reset_index(drop=True)

    return df


def create_documents_and_questions(df: pd.DataFrame):
    """
    Splits the context into snippet-level documents, and collects question info.
    
    Returns:
        all_context_docs (list[dict]): each has { 'doc_id', 'pubid', 'text' }
        questions_data (list[dict]): each has { 'pubid', 'question' }
    """
    all_context_docs = []
    questions_data = []

    for _, row in df.iterrows():
        pubid = row['pubid']
        question_text = row['question']
        context_snippets = row['context']['contexts']

        # store question
        questions_data.append({
            'pubid': pubid,
            'question': question_text
        })

        # break out each snippet
        for i, snippet in enumerate(context_snippets):
            doc_id = f"{pubid}_{i}"
            all_context_docs.append({
                'doc_id': doc_id,
                'pubid': pubid,
                'text': snippet
            })

    return all_context_docs, questions_data
