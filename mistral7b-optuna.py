import os
import json
from typing import Dict

import torch
import optuna
from datasets import Dataset

# bitsandbytes + Transformers
import bitsandbytes as bnb
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig
)

# PEFT (LoRA)
from peft import LoraConfig, get_peft_model, TaskType

# UnSloth for high-level training interface
from unsloth.trainer import UnslothTrainer, UnslothTrainingArguments


def load_jsonl_dataset(jsonl_path: str) -> Dataset:
    """
    Loads a JSONL file and returns a HuggingFace Dataset object.
    Each line in the JSONL should be a valid JSON object:
        {"instruction": "...", "response": "..."}
    """
    data_list = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data_list.append(json.loads(line))
    return Dataset.from_list(data_list)


def preprocess_function(examples: Dict[str, str], tokenizer, max_length: int = 512):
    """
    Tokenize the instruction/response pairs.
    Adjust according to your data structure.
    """
    texts = []
    for instruction, response in zip(examples["instruction"], examples["response"]):
        # Customize how you combine instruction & response:
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
    # =========================================================================
    # 1. Configurations (paths, model IDs, etc.)
    # =========================================================================

    # Path to your JSONL dataset
    train_jsonl = "datasets/theo-train.jsonl"
    eval_jsonl = "datasets/theo-eval.jsonl"  # optional

    # Choose the Mistral model
    model_id = "mistralai/Mistral-7B-v0.3"

    # Output directory for your fine-tuned model
    output_dir = "./mistral-finetuned"

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

    # =========================================================================
    # 2. BitsAndBytes quantization config
    #    (4-bit; change to load_in_8bit=True if you prefer 8-bit)
    # =========================================================================
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,                     # switch to load_in_8bit=True for 8-bit
        bnb_4bit_compute_dtype=torch.bfloat16, # keep bf16 for compute
        bnb_4bit_use_double_quant=True,        
        bnb_4bit_quant_type="nf4"              # can be "nf4" or "fp4"
    )

    # =========================================================================
    # 3. Define the Optuna objective
    # =========================================================================

    def objective(trial: optuna.trial.Trial):
        """
        Optuna objective function that:
          1. Samples hyperparameters from trial.
          2. Initializes model + LoRA adapters.
          3. Trains the model.
          4. Returns the eval_loss for the trial.
        """

        # ----------------------
        # Hyperparameter search
        # ----------------------
        num_train_epochs = trial.suggest_int("num_train_epochs", 3, 12)
        learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True)
        per_device_train_batch_size = trial.suggest_int("per_device_train_batch_size", 2, 4)
        gradient_accumulation_steps = trial.suggest_int("gradient_accumulation_steps", 1, 3)

        # LoRA hyperparams for demonstration
        lora_r = trial.suggest_int("lora_r", 4, 16)
        lora_alpha = trial.suggest_int("lora_alpha", 16, 64, step=16)
        lora_dropout = trial.suggest_float("lora_dropout", 0.0, 0.1, step=0.01)

        training_args = UnslothTrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=gradient_accumulation_steps,
            evaluation_strategy="epoch",  # Evaluate at the end of each epoch
            logging_steps=50,
            learning_rate=learning_rate,
            fp16=False,    # Turn off fp16
            bf16=True,     # Use bf16 for forward pass
        )

        # ----------------------
        # Model Initialization
        # ----------------------
        # 1) Load base model in 4-bit (or 8-bit)
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            device_map="auto",
            quantization_config=quant_config
        )

        # 2) Setup LoRA config
        #    We'll fine-tune only the LoRA layers, the rest remains in 4/8-bit
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "v_proj"]  # typical for LLMs
        )

        # 3) Attach LoRA adapters
        model = get_peft_model(base_model, lora_config)

        # ----------------------
        # Trainer setup
        # ----------------------
        trainer = UnslothTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset if eval_dataset else None,
        )

        # ----------------------
        # Training
        # ----------------------
        trainer.train()

        # ----------------------
        # Evaluation
        # ----------------------
        if eval_dataset:
            eval_results = trainer.evaluate()
            eval_loss = eval_results["eval_loss"]
            print(f"Trial {trial.number} finished with eval_loss={eval_loss:.4f}")
            return eval_loss
        else:
            train_results = trainer.evaluate(train_dataset)
            train_loss = train_results["eval_loss"]
            print(f"Trial {trial.number} finished with train_loss={train_loss:.4f} (no eval set)")
            return train_loss

    # =========================================================================
    # 4. Run the Optuna Study
    # =========================================================================
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=3)

    print("\n===== Hyperparameter Search Complete! =====")
    print(f"Best Trial ID: {study.best_trial.number}")
    print(f"Best Trial Params: {study.best_trial.params}")
    print(f"Best Trial Value (eval_loss): {study.best_value}")

    # =========================================================================
    # 5. (Optional) Final training with the best hyperparameters
    # =========================================================================

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

    # Re-initialize the base model in quantized form
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        device_map="auto",
        quantization_config=quant_config
    )

    # Create the same LoRA config
    final_lora_config = LoraConfig(
        r=final_lora_r,
        lora_alpha=final_lora_alpha,
        lora_dropout=final_lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "v_proj"]
    )
    best_model = get_peft_model(base_model, final_lora_config)

    # Final trainer
    final_trainer = UnslothTrainer(
        model=best_model,
        args=final_training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Train with best hyperparameters
    final_trainer.train()

    # Evaluate & save
    if eval_dataset:
        final_eval = final_trainer.evaluate()
        print("Final eval_loss after best hyperparam training:", final_eval["eval_loss"])

    # Saving LoRA-adapted model:
    # By default, `save_model` with a PEFT model
    # will store only the adapter weights (plus a config) in output_dir/adapter_model.bin
    final_trainer.save_model(final_training_args.output_dir)
    tokenizer.save_pretrained(final_training_args.output_dir)

    print(f"Best model training complete. Model saved to {final_training_args.output_dir}")


if __name__ == "__main__":
    main()
