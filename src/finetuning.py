# src/finetuning.py

import os
import pickle
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

from src.config import Config
from src.data import build_mnr_dataset, load_training_data, get_similarity_evaluator
import torch

def mnr_loss_finetuning(cfg: Config):
    """
    Fine-tunes a SentenceTransformer model on PubMedQA data using 
    Multiple Negatives Ranking (MNR) loss.
    Steps:
      1) Load train DataFrame
      2) Create MNR InputExamples
      3) Initialize model + loss
      4) Train
      5) Save model
    """

    # 1) Load train DataFrame from pickle
    train_df = load_training_data(cfg)
    
    print(f"Loaded training data: {len(train_df)} rows")

      # 2) Convert to HF Dataset
    train_dataset = build_mnr_dataset(train_df)
    print(f"HF Dataset columns: {train_dataset.column_names}")
    print(f"Total training examples: {len(train_dataset)}")

    # 3) Initialize model + loss
    #    We'll start with the pretrained model from cfg.MODEL_NAME
    model = SentenceTransformer(cfg.MODEL_NAME, device=cfg.DEVICE)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # 4) Define training args
    training_args = SentenceTransformerTrainingArguments(
        output_dir=cfg.MODELS_OUTPUT_DIR,
        num_train_epochs=cfg.EPOCHS,
        per_device_train_batch_size=cfg.BATCH_SIZE,
        per_device_eval_batch_size=cfg.BATCH_SIZE,
        warmup_steps=cfg.WARMUP_STEPS,
        fp16=torch.cuda.is_available(),  # use FP16 if on GPU
        eval_steps=100,                 # if you have no evaluator right now
        logging_steps=cfg.LOGGING_STEPS,
        save_steps=0,                    # we'll save at the end
        save_total_limit=1,             # keep only 1 checkpoint
    )

    # No evaluator is provided here, but you could build one if you want
    # (e.g., an EmbeddingSimilarityEvaluator on a dev set).
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=train_loss,
        evaluator = get_similarity_evaluator(cfg)
    )

    # 5) Train
    trainer.train()

    # 6) Save model
    #    This will save to the dir in cfg.MODELS_OUTPUT_DIR
    print(f"Saving fine-tuned model to: {cfg.MODELS_OUTPUT_DIR}")
    model.save(cfg.MODELS_OUTPUT_DIR)

    print("Fine-tuning complete.")
    
    return model


 # src/fine_tuning.py (Add this function)

def cosine_loss_finetuning(cfg: Config):
    """
    Fine-tunes a SentenceTransformer model using CosineSimilarityLoss.
    1) Load train DataFrame
    2) Build binary dataset with pos/neg pairs
    3) Train with CosineSimilarityLoss
    4) Save model
    """
   
    from src.data import build_binary_dataset, load_training_data

    # 1) Load
    train_df = load_training_data(cfg)
    print(f"Loaded training data: {len(train_df)} rows")

    # 2) Build dataset
    train_dataset = build_binary_dataset(train_df, negative_ratio=3, loss="cosine")
    print(f"HF Dataset columns: {train_dataset.column_names}")
    print(f"Total training examples: {len(train_dataset)}")

    # 3) CosineSimilarityLoss
    model = SentenceTransformer(cfg.MODEL_NAME, device=cfg.DEVICE)
    # The data is in [text1, text2, label], label in [0..1].
    train_loss = losses.CosineSimilarityLoss(model)

    training_args = SentenceTransformerTrainingArguments(
        output_dir=cfg.MODELS_OUTPUT_DIR,
        num_train_epochs=cfg.EPOCHS,
        per_device_train_batch_size=cfg.BATCH_SIZE,
        per_device_eval_batch_size=cfg.BATCH_SIZE,
        warmup_steps=cfg.WARMUP_STEPS,
        fp16=torch.cuda.is_available(),
        logging_steps=cfg.LOGGING_STEPS,
        save_steps=0,
        save_total_limit=1,
        eval_steps=100,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=train_loss,
        evaluator = get_similarity_evaluator(cfg)
        
    )

    # 4) Train
    trainer.train()

    print(f"Saving fine-tuned model to: {cfg.MODELS_OUTPUT_DIR}")
    model.save(cfg.MODELS_OUTPUT_DIR)
    print("Cosine Similarity fine-tuning complete.")
    return model



def softmax_loss_finetuning(cfg: Config):
    """
    Fine-tunes a SentenceTransformer model using SoftmaxLoss 
    (2-class classification: pos=0, neg=1 or vice versa).
    1) Load train DataFrame
    2) Build binary dataset (pos/neg)
    3) Train with SoftmaxLoss
    4) Save model
    """
  
    from src.data import build_binary_dataset

    train_df = load_training_data(cfg)
    print(f"Loaded training data: {len(train_df)} rows")

    # same dataset with [text1, text2, label], but label is 0 or 1 for classes
    train_dataset = build_binary_dataset(train_df, negative_ratio=3, loss="softmax")
    print(f"HF Dataset columns: {train_dataset.column_names}")
    print(f"Total training examples: {len(train_dataset)}")

    # Build model + SoftmaxLoss
    model = SentenceTransformer(cfg.MODEL_NAME, device=cfg.DEVICE)
    train_loss = losses.SoftmaxLoss(
        model=model,
        sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
        num_labels=2  # binary classification
    )

    training_args = SentenceTransformerTrainingArguments(
        output_dir=cfg.MODELS_OUTPUT_DIR,
        num_train_epochs=cfg.EPOCHS,
        per_device_train_batch_size=cfg.BATCH_SIZE,
        per_device_eval_batch_size=cfg.BATCH_SIZE,
        warmup_steps=cfg.WARMUP_STEPS,
        fp16=torch.cuda.is_available(),
        logging_steps=cfg.LOGGING_STEPS,
        save_steps=0,
        save_total_limit=1,
        eval_steps=100,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=train_loss,
        evaluator = get_similarity_evaluator(cfg)
    )

    trainer.train()

    print(f"Saving fine-tuned model to: {cfg.MODELS_OUTPUT_DIR}")
    model.save(cfg.MODELS_OUTPUT_DIR)
    print("Softmax Loss fine-tuning complete.")
    return model


