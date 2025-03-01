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


  

def train_val_test_split_pubmedqa(cfg, random_seed=42):
    """
    Splits the PubMedQA DataFrame into train, val, and test sets:
      - train: cfg.TRAIN_SIZE
      - val:   cfg.VAL_SIZE
      - test:  cfg.TEST_SIZE

    Saves the splits to files if not already present, and returns them.
    """
    # 1) Check if the split files exist
    if (
        os.path.exists(cfg.TRAIN_SPLIT_PATH)
        and os.path.exists(cfg.VAL_SPLIT_PATH)
        and os.path.exists(cfg.TEST_SPLIT_PATH)
    ):
        train_df = pd.read_pickle(cfg.TRAIN_SPLIT_PATH)
        val_df = pd.read_pickle(cfg.VAL_SPLIT_PATH)
        test_df = pd.read_pickle(cfg.TEST_SPLIT_PATH)
        print("Loaded train/val/test splits from files.")
        return train_df, val_df, test_df

    # 2) Load and shuffle data
    df = load_and_filter_data(cfg)
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    # 3) Ensure we have enough rows
    total_required = cfg.TRAIN_SIZE + cfg.VAL_SIZE + cfg.TEST_SIZE
    if len(df) < total_required:
        raise ValueError(
            f"Not enough data rows: need {total_required}, found {len(df)}."
        )

    # 4) Slice out exactly what we need
    train_df = df.iloc[: cfg.TRAIN_SIZE]
    val_df = df.iloc[cfg.TRAIN_SIZE : cfg.TRAIN_SIZE + cfg.VAL_SIZE]
    test_df = df.iloc[
        cfg.TRAIN_SIZE + cfg.VAL_SIZE : cfg.TRAIN_SIZE + cfg.VAL_SIZE + cfg.TEST_SIZE
    ]

    # 5) Ensure directories exist
    os.makedirs(os.path.dirname(cfg.TRAIN_SPLIT_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(cfg.VAL_SPLIT_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(cfg.TEST_SPLIT_PATH), exist_ok=True)

    # 6) Save splits
    train_df.to_pickle(cfg.TRAIN_SPLIT_PATH)
    val_df.to_pickle(cfg.VAL_SPLIT_PATH)
    test_df.to_pickle(cfg.TEST_SPLIT_PATH)
    print("Saved train, val, and test splits to files.")

    return train_df, val_df, test_df


def load_testing_data(cfg):
    """Loads a pre-saved train or test split from a pickle file."""
    
    # Check if the file exists in cfg.TEST_SPLIT_PATH, load it if it does, otherwise create it using load_and_filter_data
    # Return a DataFrame
    if os.path.exists(cfg.TEST_SPLIT_PATH):
        with open(cfg.TEST_SPLIT_PATH, 'rb') as f:
            return pickle.load(f)
    else:
        _, _, test_df = train_val_test_split_pubmedqa(cfg)
        return test_df


def load_training_data(cfg):
    """Loads a pre-saved train split from a pickle file."""
    
    # Check if the file exists in cfg.TRAIN_SPLIT_PATH, load it if it does, otherwise create it using load_and_filter_data
    # Return a DataFrame
    if os.path.exists(cfg.TRAIN_SPLIT_PATH):
        with open(cfg.TRAIN_SPLIT_PATH, 'rb') as f:
            return pickle.load(f)
    else:
        train_df, _, _ = train_val_test_split_pubmedqa(cfg)
        return train_df
    
def load_validation_data(cfg):
    """Loads a pre-saved val split from a pickle file."""
    

    # Check if the file exists in cfg.VAL_SPLIT_PATH, load it if it does, otherwise create it using load_and_filter_data
    # Return a DataFrame
    if os.path.exists(cfg.VAL_SPLIT_PATH):
        with open(cfg.VAL_SPLIT_PATH, 'rb') as f:
            return pickle.load(f)
    else:
        _, val_df, _ = train_val_test_split_pubmedqa(cfg)
        return val_df
    


def prepare_evaluation_data(cfg, negative_ratio=1):
    """
    Prepares an evaluation dataset (question, context, score) like STS-B:
      - Positive pairs: (question, snippet) with score=1
      - Negative pairs: (question, random snippet from another row) with score=0
    Saves the final DataFrame to cfg.EVAL_DATA_PATH.
    """
    from .data import load_validation_data
    import random

    # 1) Load validation split
    val_df = load_validation_data(cfg)

    # 2) Gather all snippets for random negatives
    all_contexts = []
    for _, row in val_df.iterrows():
        all_contexts.extend(row["context"]["contexts"])
    all_contexts = list(set(all_contexts))  # remove duplicates

    # 3) Build the evaluation pairs
    sentences1, sentences2, scores = [], [], []
    for _, row in val_df.iterrows():
        question = row["question"]
        pos_snippets = row["context"]["contexts"]

        # Positive pairs
        for snippet in pos_snippets:
            sentences1.append(question)
            sentences2.append(snippet)
            scores.append(1.0)

            # Negative pairs
            for _ in range(negative_ratio):
                neg_snippet = random.choice(all_contexts)
                while neg_snippet in pos_snippets and len(all_contexts) > len(pos_snippets):
                    neg_snippet = random.choice(all_contexts)
                sentences1.append(question)
                sentences2.append(neg_snippet)
                scores.append(0.0)

    # 4) Save to a DataFrame
    import pandas as pd
    eval_df = pd.DataFrame({
        "sentence1": sentences1,
        "sentence2": sentences2,
        "score": scores
    })

    # 5) Ensure directory exists & save
    os.makedirs(os.path.dirname(cfg.EVAL_DATA_PATH), exist_ok=True)
    eval_df.to_pickle(cfg.EVAL_DATA_PATH)
    print(f"Saved evaluation data to {cfg.EVAL_DATA_PATH}")
    
    return eval_df


def get_similarity_evaluator(cfg):
    
    """ Checks if the file exists in cfg.EVAL_DATA_PATH, loads it if it does, otherwise creates it using prepare_evaluation_data and returns as a HF dataset.
    
    """
    
    from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
    
    if os.path.exists(cfg.EVAL_DATA_PATH):
        with open(cfg.EVAL_DATA_PATH, 'rb') as f:
            eval_df = pickle.load(f)
    else:
        eval_df = prepare_evaluation_data(cfg)
        
    
    evaluator = EmbeddingSimilarityEvaluator(
        sentences1=eval_df['sentence1'].tolist(),
        sentences2=eval_df['sentence2'].tolist(),
        scores=eval_df['score'].tolist(),
        name='PubMedQA Similarity Evaluation',
        main_similarity='cosine'
    )
    
    return evaluator
    
    
    


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
        'sentences1': text1,
        'sentences2': text2,
        'label': labels
    }

    # Convert to a huggingface Dataset
    mnr_dataset = datasets.Dataset.from_dict(dataset_dict)
    return mnr_dataset





def build_binary_dataset(train_df, negative_ratio=3, loss='softmax'):
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
            if loss == 'softmax':
                labels.append(1)  # positive
            else:
                labels.append(1.0)

            # Sample negative contexts
            for _ in range(negative_ratio):
                neg_snippet = random.choice(all_contexts)
                # Make sure not to pick an actual positive
                while neg_snippet in pos_snippets and len(all_contexts) > len(pos_snippets):
                    neg_snippet = random.choice(all_contexts)
                
                text1_list.append(question)
                text2_list.append(neg_snippet)
                if loss == 'softmax':
                    labels.append(0)  # negative
                else:
                    labels.append(0.0)
                

    dataset_dict = {
        'sentences1': text1_list,
        'sentences2': text2_list,
        'label': labels
    }

    return datasets.Dataset.from_dict(dataset_dict)


# Build TSDAE training dataset,  
# 1st, load and filter dataset
# 2nd, concatenate a question with its contexts
# concatenate all questions and contexts into a list of sentences

def build_tsdae_dataset(cfg):
    
    
    # Check if the file exists in cfg.TSDAE_TRAIN_PATH, load it if it does, otherwise create it using DenoisingAutoEncoderDataset
    # Return a HF Dataset
    
    if os.path.exists(cfg.TSDAE_TRAIN_PATH):
        with open(cfg.TSDAE_TRAIN_PATH, 'rb') as f:
            return pickle.load(f)
        
    
    from sentence_transformers.datasets import DenoisingAutoEncoderDataset

    df = load_and_filter_data(cfg)
    
    
    # concatenate each question with its contexts with a period
    
    all_sentences = []
    for _, row in df.iterrows():
        question = row['question']
        contexts = row['context']['contexts']
        all_sentences.append(question + " . " + " . ".join(contexts))
        
    
    # Generate damaged sentences
    damaged_sentences = DenoisingAutoEncoderDataset(list(set(all_sentences)))
    
    # Train dataset dict 
    
    dataset_dict = {
        "damaged_sentences": [],
        "original_sentences": []
    }
    
    for sentence in damaged_sentences:
        dataset_dict["damaged_sentences"].append(sentence.texts[0])
        dataset_dict["original_sentences"].append(sentence.texts[1])
        
    # Convert to a huggingface Dataset
    tsdae_dataset = datasets.Dataset.from_dict(dataset_dict)
    
    # save it in cfg.TSDAE_TRAIN_PATH
    os.makedirs(os.path.dirname(cfg.TSDAE_TRAIN_PATH), exist_ok=True)
    tsdae_dataset.to_pickle(cfg.TSDAE_TRAIN_PATH)
    print(f"Saved TSDAE training data to {cfg.TSDAE_TRAIN_PATH}")
    
    return tsdae_dataset