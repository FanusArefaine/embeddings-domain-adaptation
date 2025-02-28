import torch

class Config:
    """
    Configuration class to store hyperparameters, file paths, model names, etc.
    Modify these as needed for different experiments.
    """
    # Model
    MODEL_NAME = "BAAI/bge-small-en-v1.5"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # or 'cpu' if no GPU

    # Data
    DATASET_NAME = "qiaojin/PubMedQA"
    DATASET_CONFIG = "pqa_artificial"
    SPLIT_NAME = "train"
    
    # Filtering criteria
    CONTEXT_COUNT = 3
    DECISIONS = ["yes", "no"]
    SAMPLE_SIZE = 2000  # how many records to sample
    TEST_SIZE = 1000
    TRAIN_SIZE = 2000
    
    # Embedding
    NORMALIZE = True     # whether to L2-normalize embeddings
    
    # Retrieval
    TOP_K = 3
    
    MODELS_OUTPUT_DIR = "models/finetuned_mnr"
    EPOCHS = 1
    BATCH_SIZE = 32
    LOGGING_STEPS = 100
    WARMUP_STEPS = 100
    
    TRAIN_SPLIT_PATH = "data/raw/pubmedqa_train.pkl"
    TEST_SPLIT_PATH = "data/raw/pubmedqa_test.pkl"
    
    TEST_CONTEXTS_PATH = "data/processed/test_contexts.pkl"
    TEST_QUESTIONS_PATH = "data/processed/test_questions.pkl"
