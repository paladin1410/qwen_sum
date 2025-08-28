import os
import yaml
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

os.environ["TRANSFORMERS_VERBOSITY"] = "error"  
transformers.logging.set_verbosity_error()

# This is the same helper function from your evaluate.py to ensure identical processing
def smart_truncate_prompt(prompt, tokenizer, max_length):
    """Apply same truncation logic as training to evaluation prompts."""
    full_tokens = tokenizer(prompt, add_special_tokens=False, truncation=False)['input_ids']
    
    if len(full_tokens) <= max_length:
        return prompt
    
    excess = len(full_tokens) - max_length
    article_start_marker = "Summarize the following news article:\n\n"
    article_end_marker = "\n\nSummary:"
    
    start_idx = prompt.find(article_start_marker) + len(article_start_marker)
    end_idx = prompt.find(article_end_marker)
    
    if start_idx != -1 and end_idx != -1:
        prefix = prompt[:start_idx]
        article = prompt[start_idx:end_idx]
        suffix = prompt[end_idx:]
        
        article_tokens = tokenizer(article, add_special_tokens=False)['input_ids']
        if len(article_tokens) > excess:
            truncated_article_tokens = article_tokens[:-excess]
            truncated_article = tokenizer.decode(truncated_article_tokens, skip_special_tokens=True)
            truncated_prompt = prefix + truncated_article + suffix
            return truncated_prompt
        else:
            fallback_tokens = full_tokens[:-excess]
            return tokenizer.decode(fallback_tokens, skip_special_tokens=True)
    else:
        fallback_tokens = full_tokens[:-excess]
        return tokenizer.decode(fallback_tokens, skip_special_tokens=True)

def extract_generated_summary(full_output, original_input_length, tokenizer, outputs):
    """Extract only the generated summary from the full model output."""
    # Method 1: Split by template marker
    if "[/INST]" in full_output:
        generated_part = full_output.split("[/INST]")[-1].strip()
    else:
        # Method 2: Extract tokens beyond original input length
        generated_tokens = outputs[0][original_input_length:]
        generated_part = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    
    return generated_part

def main():
    """
    Runs a single inference example to show the exact model input and raw output.
    """
    # Load configuration from the same YAML file
    with open('configs/param.yaml', 'r') as f:
        config = yaml.safe_load(f)

    eval_config = config['evaluation']
    
    # Define the sample article to be summarized
    sample_article = """SINGAPORE: Shoplifting was once again the top physical crime in Singapore, with 2,097 cases reported 
                        in the first half of 2025, the police said on Tuesday (Aug 26). This was a 4.2 per cent increase - 
                        or 84 cases - compared to the 2,013 cases last year from January to June. The overall number of 
                        physical crime cases similarly increased by 5.4 per cent to 10,341 in the first half of 2025, 
                        up from 9,809, according to mid-year statistics released by the Singapore Police Force (SPF). 
                        Shop theft accounted for 20.3 per cent of total crime, forming the largest proportion of cases. 
                        It also remains one of the top offences committed by youths."""

    # Load the tokenizer and model with the trained adapter
    base_path = config['model']['base_path']
    adapter_path = eval_config['adapter_path_to_evaluate']

    print(f"Loading base model from: {base_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    base_model = AutoModelForCausalLM.from_pretrained(
        base_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True, local_files_only=True
    )
    
    print(f"Loading and merging adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload()
    model.eval()
    print("Model and adapter loaded successfully.")

    # Construct the prompt using the exact template from the config
    prompt_template = config['prompt']['template']
    raw_prompt = prompt_template.format(article=sample_article)
    
    # Apply the same smart truncation as in the evaluation script
    input_max_length = eval_config['input_max_length']
    final_prompt = smart_truncate_prompt(raw_prompt, tokenizer, input_max_length)

    # Print the original input that will be fed into the model
    print("\n" + "="*25 + " FINAL MODEL INPUT " + "="*25)
    print(final_prompt)
    print("="*70 + "\n")

    # Tokenize the input and generate the output
    inputs = tokenizer(final_prompt, return_tensors="pt").to(model.device)
    generation_params = eval_config['generation']
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            **generation_params,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # Extract full output and generated summary only
    raw_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_summary = extract_generated_summary(raw_output, len(inputs['input_ids'][0]), tokenizer, outputs)

    print("="*26 + " RAW MODEL OUTPUT " + "="*26)
    print(raw_output)
    print("="*70)

    print("="*23 + " GENERATED SUMMARY ONLY " + "="*23)
    print(generated_summary)
    print("="*70)


if __name__ == "__main__":
    main()