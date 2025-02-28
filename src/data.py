# src/data.py
import os 
import pickle 
import sys
import pandas as pd
from datasets import load_dataset
from sentence_transformers import InputExample
from src.config import Config
import datasets
import random

sys.path.append('../')

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
            
    # Ensure directory exists
    os.makedirs(os.path.dirname(cfg.TEST_CONTEXTS_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(cfg.TEST_QUESTIONS_PATH), exist_ok=True)
            
    # save test all_context_docs and questions_data to pickle file (processsed)
    with open(cfg.TEST_CONTEXTS_PATH, 'wb') as f:
        pickle.dump(all_context_docs, f)
    
    with open(cfg.TEST_QUESTIONS_PATH, 'wb') as f:
        pickle.dump(questions_data, f)

    return all_context_docs, questions_data


  
def train_test_split_pubmedqa(cfg, random_seed=42):
    """
    Splits the entire PubMedQA DataFrame into train_df and test_df.
    We sample `test_size` rows for testing, the rest is training.
    Saves the splits to files and loads them if they exist.

    Args:
        cfg: Configuration object
        random_seed: Random seed for reproducibility

    Returns:
        train_df, test_df (pd.DataFrame, pd.DataFrame)
    """

    # Check if the split files exist
    if os.path.exists(cfg.TRAIN_SPLIT_PATH) and os.path.exists(cfg.TEST_SPLIT_PATH):
        # Load the dataframes from the pickle files
        train_df = pd.read_pickle(cfg.TRAIN_SPLIT_PATH)
        test_df = pd.read_pickle(cfg.TEST_SPLIT_PATH)
        print("Loaded train and test splits from files.")
        return train_df, test_df

    # Load and filter the dataset
    filter_df = load_and_filter_data(cfg)

    # Shuffle the DataFrame
    filter_df = filter_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    # Split into train and test (uisng train size and test size)
    train_df = filter_df.iloc[cfg.TEST_SIZE:]
    test_df = filter_df.iloc[:cfg.TEST_SIZE]
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(cfg.TRAIN_SPLIT_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(cfg.TEST_SPLIT_PATH), exist_ok=True)

    # Save the dataframes to pickle files
    train_df.to_pickle(cfg.TRAIN_SPLIT_PATH)
    test_df.to_pickle(cfg.TEST_SPLIT_PATH)
    print("Saved train and test splits to files.")

    return train_df, test_df



def load_testing_data(cfg):
    """Loads a pre-saved train or test split from a pickle file."""
    
    # Check if the file exists in cfg.TEST_SPLIT_PATH, load it if it does, otherwise create it using load_and_filter_data
    # Return a DataFrame
    if os.path.exists(cfg.TEST_SPLIT_PATH):
        with open(cfg.TEST_SPLIT_PATH, 'rb') as f:
            return pickle.load(f)
    else:
        _, test_df = train_test_split_pubmedqa(cfg)
        return test_df


def load_training_data(cfg):
    """Loads a pre-saved train split from a pickle file."""
    
    # Check if the file exists in cfg.TRAIN_SPLIT_PATH, load it if it does, otherwise create it using load_and_filter_data
    # Return a DataFrame
    if os.path.exists(cfg.TRAIN_SPLIT_PATH):
        with open(cfg.TRAIN_SPLIT_PATH, 'rb') as f:
            return pickle.load(f)
    else:
        train_df, _ = train_test_split_pubmedqa(cfg)
        return train_df
    
    
def build_mnr_dataset(train_df):
    """
    Creates a Hugging Face Dataset for MNR fine-tuning.
    Each row in train_df has (question, context['contexts'], final_decision, etc.).
    We'll create 3 examples for each row if each snippet is valid.
    
    Returns an HF Dataset with columns: ['text1', 'text2', 'label'].
      - text1 is the question
      - text2 is the context snippet
      - label is always 1.0 (since MNR will treat in-batch as negative)
    """

    text1, text2, labels = [], [], []

    for _, row in train_df.iterrows():
        question = row['question']
        snippets = row['context']['contexts']  # list of 3 text snippets
        # If all 3 are correct for PubMedQA, we treat each snippet as a positive pair.
        for snippet in snippets:
            text1.append(question)
            text2.append(snippet)
            labels.append(1.0)  # MNR uses in-batch negatives automatically

    dataset_dict = {
        'text1': text1,
        'text2': text2,
        'label': labels
    }

    # Convert to a huggingface Dataset
    mnr_dataset = datasets.Dataset.from_dict(dataset_dict)
    return mnr_dataset





def build_binary_dataset(train_df, negative_ratio=1):
    """
    Creates a Hugging Face Dataset of pairs (text1, text2, label)
    for binary classification or similarity tasks.
    
    - We assume each row in train_df has:
       question, context['contexts'] (one or multiple positives)
    - We randomly sample `negative_ratio` negative contexts for each positive.
    
    Returns an HF Dataset with columns: ['text1', 'text2', 'label'].
      label=1.0 for positive, label=0.0 for negative
    """
    text1_list = []
    text2_list = []
    labels = []

    # We'll gather *all* possible contexts in a big list for random sampling as negatives
    all_contexts = []
    for _, row in train_df.iterrows():
        all_contexts.extend(row['context']['contexts'])
    all_contexts = list(set(all_contexts))  # remove duplicates

    for _, row in train_df.iterrows():
        question = row['question']
        pos_snippets = row['context']['contexts']

        # For each snippet, make a positive pair
        for snippet in pos_snippets:
            text1_list.append(question)
            text2_list.append(snippet)
            labels.append(1.0)  # positive

            # Sample negative contexts
            for _ in range(negative_ratio):
                neg_snippet = random.choice(all_contexts)
                # Make sure not to pick an actual positive
                while neg_snippet in pos_snippets and len(all_contexts) > len(pos_snippets):
                    neg_snippet = random.choice(all_contexts)
                
                text1_list.append(question)
                text2_list.append(neg_snippet)
                labels.append(0.0)  # negative

    dataset_dict = {
        'text1': text1_list,
        'text2': text2_list,
        'label': labels
    }

    return datasets.Dataset.from_dict(dataset_dict)
