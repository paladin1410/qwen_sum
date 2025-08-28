import os
import yaml
import torch
import numpy as np
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_dataset
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from src.data.processing import CustomDataCollator

os.environ["TRANSFORMERS_VERBOSITY"] = "error"  
transformers.logging.set_verbosity_error()


def load_models(config):
    """Load base model and adapter if specified in the config."""
    base_path = config['model']['base_path']
    adapter_path = config['evaluation'].get('adapter_path_to_evaluate')
    if not adapter_path:
        adapter_path = config['training']['adapter_save_path']
        print(f"'adapter_path_to_evaluate' not set. Defaulting to training path: {adapter_path}")

    print(f"Loading base model from: {base_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Set same settings as training
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    tokenizer.model_max_length = config['model']['max_length']
        
    # Load base model
    base_model = AutoModelForCausalLM.from_pretrained(
        base_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True, local_files_only=True
    )
    
    adapter_model = None
    if os.path.exists(adapter_path):
        print(f"Loading adapter from: {adapter_path}")
        
        # Load a SEPARATE base model for adapter
        adapter_base = AutoModelForCausalLM.from_pretrained(
            base_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True, local_files_only=True
        )
        
        adapter_model = PeftModel.from_pretrained(adapter_base, adapter_path)
        adapter_model = adapter_model.merge_and_unload()
        print("✅ Adapter loaded and merged successfully.")
    else:
        print(f"⚠️ No adapter found at: {adapter_path}. Evaluating base model only.")

    return base_model, adapter_model, tokenizer

def smart_truncate_prompt(prompt, tokenizer, max_length):
    """Apply same truncation logic as training to evaluation prompts."""
    # Tokenize the full prompt first
    full_tokens = tokenizer(prompt, add_special_tokens=False, truncation=False)['input_ids']
    
    if len(full_tokens) <= max_length:
        return prompt
    
    # Apply same smart truncation as training
    excess = len(full_tokens) - max_length
    article_start_marker = "Summarize the following news article:\n\n"
    article_end_marker = "\n\nSummary:"
    
    start_idx = prompt.find(article_start_marker) + len(article_start_marker)
    end_idx = prompt.find(article_end_marker)
    
    if start_idx != -1 and end_idx != -1:
        # Extract parts
        prefix = prompt[:start_idx]  # "[INST] Summarize the following news article:\n\n"
        article = prompt[start_idx:end_idx]  # The actual article content
        suffix = prompt[end_idx:]  # "\n\nSummary: [/INST]"
        
        # Truncate only the article part
        article_tokens = tokenizer(article, add_special_tokens=False)['input_ids']
        if len(article_tokens) > excess:
            truncated_article_tokens = article_tokens[:-excess]  # Remove from end of article
            truncated_article = tokenizer.decode(truncated_article_tokens, skip_special_tokens=True)
            
            # Reconstruct with truncated article
            truncated_prompt = prefix + truncated_article + suffix
            return truncated_prompt
        else:
            # Fallback if article is shorter than excess
            fallback_tokens = full_tokens[:-excess]
            return tokenizer.decode(fallback_tokens, skip_special_tokens=True)
    else:
        # Fallback truncation
        fallback_tokens = full_tokens[:-excess]
        return tokenizer.decode(fallback_tokens, skip_special_tokens=True)

def evaluate_model(model, tokenizer, dataset, model_name, eval_config, prompt_template):
    """Evaluate the model using batch inference with parameters from the config."""

    # Use eval_template if available, otherwise fall back to training template
    # eval_template = eval_config.get('eval_template', prompt_template)

    num_examples = eval_config['num_examples']
    batch_size = eval_config['batch_size']
    input_max_length = eval_config['input_max_length']
    generation_params = eval_config['generation']

    print(f"🚀 Evaluating {model_name} on {num_examples} examples...")
    
    actual_examples = min(num_examples, len(dataset))
    summaries, references = [], []
    
    model.eval()
    with torch.no_grad():
        for i in range(0, actual_examples, batch_size):

            end_index = min(i + batch_size, actual_examples)
            batch_data = dataset[i:end_index]
            
            if isinstance(batch_data['article'], str):
                batch_articles = [batch_data['article']]
                batch_references = [batch_data['highlights']]
            else:
                batch_articles = batch_data['article']
                batch_references = batch_data['highlights']

            references.extend(batch_references)
            
            # Create prompts using same template as training
            raw_prompts = [prompt_template.format(article=article) for article in batch_articles]
            # Create prompts using eval template
            # raw_prompts = [eval_template.format(article=article) for article in batch_articles]

            # Apply same smart truncation as training
            truncated_prompts = [smart_truncate_prompt(prompt, tokenizer, input_max_length) for prompt in raw_prompts]

            # Tokenize using same settings as training
            inputs = tokenizer(
                truncated_prompts, 
                return_tensors="pt", 
                padding=True, 
                truncation=False,  # Same as training
                add_special_tokens=False  # Same as training
            ).to(model.device)

            outputs = model.generate(
                **inputs, 
                **generation_params,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
            # Decode generated outputs
            batch_summaries = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            # Extract only the generated part (after [/INST])
            cleaned_summaries = []
            for j, full_output in enumerate(batch_summaries):
                # print(f"  Full output:  {full_output}")
                if "[/INST]" in full_output:
                    generated_part = full_output.split("[/INST]")[-1].strip()
                else:
                    # Fallback: try to extract from the end
                    original_length = len(inputs['input_ids'][j])
                    generated_tokens = outputs[j][original_length:]
                    generated_part = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                
                cleaned_summaries.append(generated_part)

            summaries.extend(cleaned_summaries)
            
            if (i + batch_size) % 50 == 0 or end_index == actual_examples:
                print(f"  Processed {len(summaries)}/{actual_examples} examples...")


    print("  Computing ROUGE-L and BERTScore...")
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_scores = [scorer.score(ref, summ)['rougeL'].fmeasure for ref, summ in zip(references, summaries)]
    
    try:
        _, _, F1 = bert_score(summaries, references, lang="en", verbose=False, batch_size=min(batch_size, 64))
        bert_f1 = F1.mean().item()
    except Exception as e:
        print(f"  BERTScore error: {e}")
        bert_f1 = 0.0

    return {
        'model': model_name,
        'rouge_l': np.mean(rouge_scores),
        'bert_f1': bert_f1,
        'avg_gen_len': np.mean([len(s.split()) for s in summaries]),
        'avg_ref_len': np.mean([len(r.split()) for r in references]),
        'num_examples': len(summaries)
    }

def print_results(results, base_results=None):
    """Helper function to print evaluation metrics."""
    print(f"\n{'📊' if base_results is None else '🎯'} {results['model']} ({results['num_examples']} examples):")
    print(f"  ROUGE-L: {results['rouge_l']:.4f}")
    print(f"  BERTScore F1: {results['bert_f1']:.4f}")
    print(f"  Avg Length: {results['avg_gen_len']:.1f} words (ref: {results['avg_ref_len']:.1f})")
    
    if base_results:
        rouge_diff = results['rouge_l'] - base_results['rouge_l']
        bert_diff = results['bert_f1'] - base_results['bert_f1']
        print(f"\n📈 Improvement:")
        print(f"  ROUGE-L: {rouge_diff:+.4f}")
        print(f"  BERTScore F1: {bert_diff:+.4f}")

def main():
    with open('configs/param.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    base_model, adapter_model, tokenizer = load_models(config)
    
    dataset = load_dataset(
        config['dataset']['name'], 
        config['dataset']['version'], 
        split=config['dataset']['test_split']
    )

    print("\n" + "="*60 + "\nBASELINE EVALUATION - ROUGE-L & BERTScore\n" + "="*60)
    
    eval_config = config['evaluation']
    prompt_template = config['prompt']['template']

    base_results = None
    base_results = evaluate_model(base_model, tokenizer, dataset, "Base Model", eval_config, prompt_template)
    
    adapter_results = None
    if adapter_model is not None:
        adapter_results = evaluate_model(adapter_model, tokenizer, dataset, "LoRA Adapter", eval_config, prompt_template)

    print("\n" + "="*60 + "\nRESULTS\n" + "="*60)
    if base_results:
        print_results(base_results)
    if adapter_results:
        print_results(adapter_results, base_results)
    

if __name__ == "__main__":
    main()
