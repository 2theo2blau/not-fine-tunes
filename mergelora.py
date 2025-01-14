import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

model_id = "mistralai/Mistral-7B-v0.3"

# If you're quantizing (4-bit example):
# quant_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_compute_dtype=torch.bfloat16,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4"
# )

# 1. Load your base model
base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=None,  # if you used 4-bit/8-bit during training
    trust_remote_code=True,
)

# 2. Load the LoRA adapter
#    "my_lora_adapter_dir" is where you ran final_trainer.save_model(...) 
lora_model = PeftModel.from_pretrained(base_model, "mistral-finetuned/checkpoint-1000/")

merged_model = lora_model.merge_and_unload()

save_path = "mistral-7b-merged"
merged_model.save_pretrained(save_path)

# If you also want to save the tokenizer:
# (Often the same tokenizer used for both base and LoRA)
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
tokenizer.save_pretrained(save_path)

print(f"Merged model saved to: {save_path}")