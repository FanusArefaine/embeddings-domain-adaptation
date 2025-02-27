class Config:
    """
    Configuration class to store hyperparameters, file paths, model names, etc.
    Modify these as needed for different experiments.
    """
    # Model
    MODEL_NAME = "BAAI/bge-small-en-v1.5"
    DEVICE = "cuda"  # or 'cpu' if no GPU

    # Data
    DATASET_NAME = "qiaojin/PubMedQA"
    DATASET_CONFIG = "pqa_artificial"
    SPLIT_NAME = "train"
    
    # Filtering criteria
    CONTEXT_COUNT = 3
    DECISIONS = ["yes", "no"]
    SAMPLE_SIZE = 2000  # how many records to sample
    
    # Embedding
    NORMALIZE = True     # whether to L2-normalize embeddings
    
    # Retrieval
    TOP_K = 3
