import os
import json
from typing import Dict
import torch
import optuna
from datasets import Dataset
import bitsandbytes as bnb
from transformers import AutoTokenizer, AutoModelForCausalLM,  BitsAndBytesConfig

from peft import LoraConfig, get_peft_model, TaskType
from unsloth.trainer import UnslothTrainer, UnslothTrainingArguments


# Load JSONL dataset
def load_jsonl_dataset(jsonl_path: str) -> Dataset:
    data_list = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data_list.append(json.loads(line))
    return Dataset.from_list(data_list)

# Tokenize instruction/response pairs
def preprocess_function(examples: Dict[str, str], tokenizer, max_length: int = 512):
    texts = []
    for instruction, response in zip(examples["instruction"], examples["response"]):
        text = f"Instruction: {instruction}\n\nResponse: {response}"
        texts.append(text)

    tokenized = tokenizer(
        texts,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    return tokenized


def main():

    # Path to JSONL dataset
    train_jsonl = "datasets/theo-train.jsonl"
    eval_jsonl = "datasets/theo-eval.jsonl"  # optional

    # Specify model
    model_id = "mistralai/Mistral-Nemo-Instruct-2407"

    # Output directory
    output_dir = "./mistral-nemo-finetuned-0125"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    train_dataset = load_jsonl_dataset(train_jsonl)
    eval_dataset = None
    if os.path.exists(eval_jsonl):
        eval_dataset = load_jsonl_dataset(eval_jsonl)

    # Pre-tokenize / map the dataset
    train_dataset = train_dataset.map(
        lambda examples: preprocess_function(examples, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names
    )
    if eval_dataset is not None:
        eval_dataset = eval_dataset.map(
            lambda examples: preprocess_function(examples, tokenizer),
            batched=True,
            remove_columns=eval_dataset.column_names
        )

    # BitsAndBytes config
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,        
        bnb_4bit_quant_type="fp4"
    )

    # Define the Optuna objective
    def objective(trial: optuna.trial.Trial):
        # Create a subset of training data for faster trials
        trial_train_dataset = train_dataset.shuffle(seed=42).select(range(min(len(train_dataset), 400)))
        trial_eval_dataset = eval_dataset.shuffle(seed=42).select(range(min(len(eval_dataset), 80))) if eval_dataset else None

        # Hyperparameter search
        num_train_epochs = trial.suggest_int("num_train_epochs", 3, 6)  # Reduced from 3-6
        learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True)
        per_device_train_batch_size = trial.suggest_int("per_device_train_batch_size", 2, 4)
        gradient_accumulation_steps = trial.suggest_int("gradient_accumulation_steps", 1, 3)

        # LoRA hyperparams
        lora_r = trial.suggest_int("lora_r", 4, 16)
        lora_alpha = trial.suggest_int("lora_alpha", 16, 64, step=16)
        lora_dropout = trial.suggest_float("lora_dropout", 0.02, 0.1, step=0.01)

        training_args = UnslothTrainingArguments(
            output_dir=os.path.join(output_dir, f"trial_{trial.number}"),
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=gradient_accumulation_steps,
            evaluation_strategy="epoch",
            logging_steps=10,  # Reduced from 50
            learning_rate=learning_rate,
            fp16=False,
            bf16=True,
        )

        # Model Initialization
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            device_map="auto",
            quantization_config=quant_config
        )

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "v_proj"]
        )

        model = get_peft_model(base_model, lora_config)

        trainer = UnslothTrainer(
            model=model,
            args=training_args,
            train_dataset=trial_train_dataset,
            eval_dataset=trial_eval_dataset if trial_eval_dataset else None,
        )

        # Training
        trainer.train()

        # Evaluation
        if trial_eval_dataset:
            eval_results = trainer.evaluate()
            eval_loss = eval_results["eval_loss"]
            print(f"Trial {trial.number} finished with eval_loss={eval_loss:.4f}")
            return eval_loss
        else:
            train_results = trainer.evaluate(trial_train_dataset)
            train_loss = train_results["eval_loss"]
            print(f"Trial {trial.number} finished with train_loss={train_loss:.4f} (no eval set)")
            return train_loss

    # Run the Optuna Study
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=10)

    print("\n===== Hyperparameter Search Complete! =====")
    print(f"Best Trial ID: {study.best_trial.number}")
    print(f"Best Trial Params: {study.best_trial.params}")
    print(f"Best Trial Value (eval_loss): {study.best_value}")

    # Final training with the best hyperparameters

    best_params = study.best_trial.params

    # Extract best hyperparams
    final_num_train_epochs = best_params["num_train_epochs"]
    final_learning_rate = best_params["learning_rate"]
    final_per_device_train_batch_size = best_params["per_device_train_batch_size"]
    final_gradient_accumulation_steps = best_params["gradient_accumulation_steps"]

    # LoRA best hyperparams
    final_lora_r = best_params["lora_r"]
    final_lora_alpha = best_params["lora_alpha"]
    final_lora_dropout = best_params["lora_dropout"]

    final_training_args = UnslothTrainingArguments(
        output_dir=os.path.join(output_dir, "best_model"),
        num_train_epochs=final_num_train_epochs,
        per_device_train_batch_size=final_per_device_train_batch_size,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=final_gradient_accumulation_steps,
        evaluation_strategy="epoch",
        learning_rate=final_learning_rate,
        fp16=False,
        bf16=True,
        logging_steps=50,
    )

    # Re-initialize base model
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        device_map="auto",
        quantization_config=quant_config
    )

    # Create final LoRA config
    final_lora_config = LoraConfig(
        r=final_lora_r,
        lora_alpha=final_lora_alpha,
        lora_dropout=final_lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "v_proj"]
    )
    best_model = get_peft_model(base_model, final_lora_config)

    # Final trainer with best hyperparameters
    final_trainer = UnslothTrainer(
        model=best_model,
        args=final_training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    final_trainer.train()

    # Evaluate & save
    if eval_dataset:
        final_eval = final_trainer.evaluate()
        print("Final eval_loss after best hyperparam training:", final_eval["eval_loss"])

    # Save LoRA model
    final_trainer.save_model(final_training_args.output_dir)
    tokenizer.save_pretrained(final_training_args.output_dir)

    print(f"Best model training complete. Model saved to {final_training_args.output_dir}")


if __name__ == "__main__":
    main()
