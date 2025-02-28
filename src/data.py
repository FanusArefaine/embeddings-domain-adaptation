# src/data.py
import os 
import pickle 
import pandas as pd
from datasets import load_dataset
from sentence_transformers import InputExample


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

    return df


def get_questions_and_context_docs(cfg, df: pd.DataFrame):
    """
    Splits the context into snippet-level documents, and collects question info.
    
    Returns:
        all_context_docs (list[dict]): each has { 'doc_id', 'pubid', 'text' }
        questions_data (list[dict]): each has { 'pubid', 'question' }
    """
    
    # Check if processed files exist in cfg.TEST_CONTEXTS_PATH and cfg.TEST_QUESTIONS_PATH
    if os.path.exists(cfg.TEST_CONTEXTS_PATH) and os.path.exists(cfg.TEST_QUESTIONS_PATH):
        with open(cfg.TEST_CONTEXTS_PATH, 'rb') as f:
            all_context_docs = pickle.load(f)
        with open(cfg.TEST_QUESTIONS_PATH, 'rb') as f:
            questions_data = pickle.load(f)
        return all_context_docs, questions_data
    
    
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
            
    # save test all_context_docs and questions_data to pickle file (processsed)
    with open(cfg.TEST_CONTEXTS_PATH, 'wb') as f:
        pickle.dump(all_context_docs, f)
    
    with open(cfg.TEST_QUESTIONS_PATH, 'wb') as f:
        pickle.dump(questions_data, f)

    return all_context_docs, questions_data


  
def train_test_split_pubmedqa(full_df, test_size=2000, random_seed=42):
    """
    Splits the entire PubMedQA DataFrame into train_df and test_df.
    We sample `test_size` rows for testing, the rest is training.
    
    Args:
        full_df (pd.DataFrame): Contains all 14k questions
        test_size (int): number of rows to set aside for test
        random_seed (int): for reproducibility
    
    Returns:
        train_df, test_df (pd.DataFrame, pd.DataFrame)
    """
    if len(full_df) <= test_size:
        raise ValueError("Not enough data to split!")
    
    test_df = full_df.sample(n=test_size, random_state=random_seed)
    train_df = full_df.drop(test_df.index).reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    return train_df, test_df



def load_testing_data(cfg):
    """Loads a pre-saved train or test split from a pickle file."""
    
    # Check if the file exists in cfg.TEST_SPLIT_PATH, load it if it does, otherwise create it using load_and_filter_data
    # Return a DataFrame
    if os.path.exists(cfg.TEST_SPLIT_PATH):
        with open(cfg.TEST_SPLIT_PATH, 'rb') as f:
            return pickle.load(f)
    else:
        df = load_and_filter_data(cfg)
        with open(cfg.TEST_SPLIT_PATH, 'wb') as f:
            pickle.dump(df, f)
        return df


def load_training_data(cfg):
    """Loads a pre-saved train split from a pickle file."""
    
    if not os.path.exists(cfg.TRAIN_SPLIT_PATH):
        raise FileNotFoundError(f"Train split not found at {cfg.TRAIN_SPLIT_PATH}")
    with open(cfg.TRAIN_SPLIT_PATH, 'rb') as f:
        return pickle.load(f)
    
def build_mnr_samples(train_df):
    """
    For each row in train_df, we create InputExample(question, correct_context).
    We assume each row has exactly 3 context snippets, but only 1 is correct for MNR
    (PubMedQA in your use case might store them differently if all 3 are correct).
    Actually, for PubMedQA, all 3 might be from the same pubid, so we treat them as positives?

    But typically you'd do:
      question => context snippet #1, #2, #3 as the "positive" pairs
      

    If you have a single correct snippet, you just do (question, snippet).

    Returns a list of InputExample
    """
    # This example assumes there's only 1 "correct" snippet per question.
    # If in your data, all 3 are correct for that question, you can create 3 examples per question.
    
    samples = []
    for idx, row in train_df.iterrows():
        question = row['question']
        snippets = row['context']['contexts']  # list of text
        # In your dataset, all 3 might be relevant. If so, let's create 3 samples:
        for snippet in snippets:
            samples.append(InputExample(
                texts=[question, snippet], 
                label=1.0  # or None; MNR doesn't strictly need a label
            ))
    return samples