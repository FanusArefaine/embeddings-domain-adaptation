# src/fine_tuning.py

import os
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

def fine_tune_mnr(cfg, train_samples):
    """
    Fine-tune a Sentence Transformers model using MultipleNegativesRankingLoss on the given train samples.
    
    Args:
        cfg (Config): Your config object with model details, device, etc.
        train_samples (list of InputExample): MNR training data
    
    Returns:
        model (SentenceTransformer): the fine-tuned model
    """
    # 1) Load base pretrained model
    model = SentenceTransformer(cfg.MODEL_NAME, device=cfg.DEVICE)

    # 2) MNR loss
    train_loss = losses.MultipleNegativesRankingLoss(model=model)
    
    # 3) Prepare training arguments
    args = SentenceTransformerTrainingArguments(
        output_dir=cfg.MODELS_OUTPUT_DIR,  # e.g. "models/finetuned_mnr"
        num_train_epochs=cfg.EPOCHS, 
        per_device_train_batch_size=cfg.BATCH_SIZE,
        fp16=True,
        logging_steps=cfg.LOGGING_STEPS,
        eval_steps=None,  # or set if you have a dev set
        warmup_steps=cfg.WARMUP_STEPS
    )

    # 4) Create the trainer
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_samples,
        loss=train_loss,
        # If you have an evaluator, you can pass it here
    )
    
    # 5) Train
    trainer.train()
    
    # 6) Save the model
    if not os.path.exists(cfg.MODELS_OUTPUT_DIR):
        os.makedirs(cfg.MODELS_OUTPUT_DIR, exist_ok=True)
    model.save(cfg.MODELS_OUTPUT_DIR)
    
    return model
