import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

model_id = "mistralai/Mistral-7B-Instruct-v0.3"

# Load your base model
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=None, 
    trust_remote_code=True,
)

# Load the LoRA adapter
lora_model = PeftModel.from_pretrained(base_model, "mistral-inst-finetuned-0123/best_model/")

merged_model = lora_model.merge_and_unload()

save_path = "thistral-7b-0123"
merged_model.save_pretrained(save_path)

# If you also want to save the tokenizer:
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
tokenizer.save_pretrained(save_path)

print(f"Merged model saved to: {save_path}")