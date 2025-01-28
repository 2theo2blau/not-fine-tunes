import json
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# Load model and tokenizer
def initialize_pipeline(model_name="mistralai/Mistral-7B-Instruct-v0.3"):
    print("Initializing pipeline with model:", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return generator

# Load prompts from a JSON file
def load_prompts(input_file):
    with open(input_file, "r") as file:
        data = json.load(file)
    return data["prompts"]

# Process prompts and generate responses
def process_prompts(generator, prompts, output_file):
    results = []
    
    print("Processing prompts...")
    for prompt in tqdm(prompts, desc="Generating responses"):
        response = generator(
            prompt, 
            max_length=256, 
            num_return_sequences=1,
            # temperature=0.6,  
            # top_p=0.9,       
            # top_k=40,        
            do_sample=False
        )[0]["generated_text"]
        results.append({"prompt": prompt, "response": response})

    # Write results to a JSON file
    with open(output_file, "w") as file:
        json.dump({"results": results}, file, indent=4)

    print(f"Processing complete. Results saved to {output_file}")

# Main script
def main():
    input_file = "instruction_set.json"
    output_file = "instruction_response_pairs.json"
    model_name = "mistralai/Mistral-7B-Instruct-v0.3"  

    generator = initialize_pipeline(model_name)
    prompts = load_prompts(input_file)

    process_prompts(generator, prompts, output_file)

if __name__ == "__main__":
    main()