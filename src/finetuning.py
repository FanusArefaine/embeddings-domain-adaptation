# src/finetuning.py

import os
import pickle
from sentence_transformers import SentenceTransformer, losses, models
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

from src.config import Config
from src.data import build_mnr_dataset, load_training_data, get_similarity_evaluator, build_mlm_dataset
import torch
from sentence_transformers.training_args import BatchSamplers


def mnr_loss_finetuning(cfg: Config, model=None):
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
    if model is None:
        model = SentenceTransformer(cfg.MODEL_NAME, device=cfg.DEVICE)
        
    evaluator = get_similarity_evaluator(cfg)
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
        batch_sampler=BatchSamplers.NO_DUPLICATES   #https://github.com/UKPLab/sentence-transformers/issues/2827
    )

    # No evaluator is provided here, but you could build one if you want
    # (e.g., an EmbeddingSimilarityEvaluator on a dev set).
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=train_loss,
        evaluator = evaluator
    )

    # 5) Train
    trainer.train()
    
    print("Evaluator Results:", evaluator(model))

    # 6) Save model
    #    This will save to the dir in cfg.MODELS_OUTPUT_DIR
    print(f"Saving fine-tuned model to: {cfg.MODELS_OUTPUT_DIR}")
    model.save(cfg.MODELS_OUTPUT_DIR)

    print("Fine-tuning complete.")
    
    return model


 # src/fine_tuning.py (Add this function)

def cosine_loss_finetuning(cfg: Config, model=None):
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
    if model is None:
        model = SentenceTransformer(cfg.MODEL_NAME, device=cfg.DEVICE)
    
    evaluator = get_similarity_evaluator(cfg)
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
        evaluator = evaluator
        
    )

    # 4) Train
    trainer.train()
    
    print("Evaluator Results:", evaluator(model))

    print(f"Saving fine-tuned model to: {cfg.MODELS_OUTPUT_DIR}")
    model.save(cfg.MODELS_OUTPUT_DIR)
    print("Cosine Similarity fine-tuning complete.")
    return model

def softmax_loss_finetuning(cfg: Config, model=None):
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
    if model is None:
        model = SentenceTransformer(cfg.MODEL_NAME, device=cfg.DEVICE)
        
    evaluator = get_similarity_evaluator(cfg)
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
        evaluator = evaluator
    )

    trainer.train()
    
    print("Evaluator Results:", evaluator(model))

    print(f"Saving fine-tuned model to: {cfg.MODELS_OUTPUT_DIR}")
    model.save(cfg.MODELS_OUTPUT_DIR)
    print("Softmax Loss fine-tuning complete.")
    return model



def tsdae_finetuning(cfg: Config):
    """
    Fine-tunes a SentenceTransformer model using a TSDAE objective.
    1) Build TSDAE dataset from domain data.
    2) Initialize model + TSDAE loss.
    3) Train the model.
    4) Save the model.
    """
    from src.data import build_tsdae_dataset
    from sentence_transformers import SentenceTransformer, losses, models
    from sentence_transformers.trainer import SentenceTransformerTrainer
    from sentence_transformers.training_args import SentenceTransformerTrainingArguments
    import torch
    
    import nltk
    nltk.download('punkt')
    nltk.download('punkt_tab')

    # 1) Build TSDAE dataset
    tsdae_dataset = build_tsdae_dataset(cfg)

    print(f"TSDAE dataset columns: {tsdae_dataset.column_names}")
    print(f"Total TSDAE examples: {len(tsdae_dataset)}")

    # 2) Initialize model + TSDAE loss
    default_model = models.Transformer(cfg.MODEL_NAME, max_seq_length=512)
    pooling_model = models.Pooling(default_model.get_word_embedding_dimension(), "cls")
    cls_pooled_model = SentenceTransformer(modules=[default_model, pooling_model])

    evaluator = get_similarity_evaluator(cfg)
    
    train_loss = losses.DenoisingAutoEncoderLoss(
        cls_pooled_model, 
        tie_encoder_decoder=True
    )
    train_loss.decoder = train_loss.decoder.to("cuda")


    # 3) Train with the sentence-transformers trainer
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
        eval_steps=100
    )

    trainer = SentenceTransformerTrainer(
        model=cls_pooled_model,
        args=training_args,
        train_dataset=tsdae_dataset,
        loss=train_loss,
        evaluator=evaluator
    )
    trainer.train()
    
    print("Evaluator Results:", evaluator(cls_pooled_model))

    # 4) Save trained model
    cls_pooled_model.save(cfg.MODELS_OUTPUT_DIR)
    print(f"TSDAE fine-tuning complete. Model saved to {cfg.MODELS_OUTPUT_DIR}")
    return cls_pooled_model



def create_st_model_from_mlm(mlm_model_path):
    """
    Creates a SentenceTransformer from a Hugging Face model directory
    (the result of masked language modeling).
    We do something like:
      - transformer = models.Transformer(mlm_model_path)
      - pooling = models.Pooling(...)
      - st_model = SentenceTransformer([transformer, pooling])
    """
    transformer_model = models.Transformer(mlm_model_path, max_seq_length=512)
    pooling_model = models.Pooling(
        transformer_model.get_word_embedding_dimension(),
        pooling_mode_cls_token=True
    )
    st_model = SentenceTransformer(
        modules=[transformer_model, pooling_model]
    )
    return st_model


# src/fine_tuning.py  (add this function)

def mlm_finetuning(cfg):
    """
    Continues pretraining for masked language modeling on domain text.
    1) Builds/loads domain text dataset (1 column: 'text').
    2) Tokenizes with AutoTokenizer.
    3) Uses DataCollatorForLanguageModeling for random masking.
    4) Trains with Hugging Face Trainer on AutoModelForMaskedLM.
    5) Saves final model to cfg.MODELS_OUTPUT_DIR.
    
    Return: the path to the saved model or the model object.
    """
    import os
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForMaskedLM,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments
    )
    from src.data import build_mlm_dataset

    # 1) Build domain dataset
    mlm_dataset = build_mlm_dataset(cfg)
    print(f"MLM Dataset columns: {mlm_dataset.column_names}, size={len(mlm_dataset)}")

    # 2) Tokenize
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)
    
    tokenized_dataset = mlm_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_dataset.set_format("torch")  # so we get PyTorch Tensors

    # We can split into 'train' / 'eval' if we want:
    # For demonstration, let's do 90% train, 10% eval
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    print(f"Train set size: {len(train_dataset)}, Eval set size: {len(eval_dataset)}")

    # 3) Create model
    model = AutoModelForMaskedLM.from_pretrained(cfg.MODEL_NAME)
    
    # 4) DataCollator for MLM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=True, 
        mlm_probability=0.15
    )

    # 5) Training args
    training_args = TrainingArguments(
        output_dir=cfg.MODELS_OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=cfg.EPOCHS,
        per_device_train_batch_size=cfg.BATCH_SIZE,
        per_device_eval_batch_size=cfg.BATCH_SIZE,
        logging_steps=cfg.LOGGING_STEPS,
        save_steps=0,  # or a certain step
        evaluation_strategy="steps" if len(eval_dataset) > 0 else "no",
        eval_steps=200,  # or any you prefer
        warmup_steps=cfg.WARMUP_STEPS,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=2,
        save_total_limit=1
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset if len(eval_dataset) > 0 else None,
        data_collator=data_collator
    )

    # 6) Train
    trainer.train()

    # Evaluate final perplexity if you want:
    if len(eval_dataset) > 0:
        eval_results = trainer.evaluate()
        print(f"Final eval results: {eval_results}")

    # 7) Save model
    trainer.save_model(cfg.MODELS_OUTPUT_DIR)
    tokenizer.save_pretrained(cfg.MODELS_OUTPUT_DIR)
    print(f"MLM continued pretraining complete. Model saved at {cfg.MODELS_OUTPUT_DIR}")
    
    model = create_st_model_from_mlm(cfg.MODELS_OUTPUT_DIR)

    return model
