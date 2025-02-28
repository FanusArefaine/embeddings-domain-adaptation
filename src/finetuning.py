# src/fine_tuning.py

import os
import pickle
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

from src.config import Config
from src.data import build_mnr_samples, load_training_data
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

    # 2) Create MNR InputExamples
    #    Using your existing helper from data.py
    train_examples = build_mnr_samples(train_df)
    print(f"Created {len(train_examples)} MNR training samples")

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
        eval_steps=None,                 # if you have no evaluator right now
        logging_steps=cfg.LOGGING_STEPS,
        save_steps=0,                    # we'll save at the end
        save_total_limit=1,             # keep only 1 checkpoint
    )

    # No evaluator is provided here, but you could build one if you want
    # (e.g., an EmbeddingSimilarityEvaluator on a dev set).
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_examples,
        loss=train_loss,
        # evaluator=...,  # Optional if you want validation
    )

    # 5) Train
    trainer.train()

    # 6) Save model
    #    This will save to the dir in cfg.MODELS_OUTPUT_DIR
    print(f"Saving fine-tuned model to: {cfg.MODELS_OUTPUT_DIR}")
    model.save(cfg.MODELS_OUTPUT_DIR)

    print("Fine-tuning complete.")
