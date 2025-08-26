# src/evaluation/evaluate.py - Fixed version without SummaC
"""
Simple baseline evaluation focusing on 2 working metrics:
1. ROUGE-L (content overlap) 
2. BERTScore F1 (semantic similarity)
"""
import os
import yaml
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_dataset
from rouge_score import rouge_scorer
from bert_score import score as bert_score



def load_models(config):
    """Load base model and adapter if exists"""
    base_path = config['model']['base_path']
    adapter_path = config['training']['adapter_save_path']
    
    # Load tokenizer and base model
    print(f"Loading base model from: {base_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    base_model = AutoModelForCausalLM.from_pretrained(
        base_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    
    # Load adapter if exists
    adapter_model = None
    if os.path.exists(adapter_path):
        print(f"Loading adapter from: {adapter_path}")
        adapter_model = PeftModel.from_pretrained(base_model, adapter_path)
        adapter_model = adapter_model.merge_and_unload()
        print("✅ Adapter loaded successfully")
    else:
        print(f"⚠️  No adapter found at: {adapter_path}")
        
    return base_model, adapter_model, tokenizer

def evaluate_model(model, tokenizer, dataset, model_name, num_examples=100, batch_size=128):
    """Evaluate model using batch inference - NO SummaC"""
    print(f"Evaluating {model_name} on {num_examples} examples...")
    
    actual_examples = min(num_examples, len(dataset))
    
    summaries = []
    references = []
    
    model.eval()
    with torch.no_grad():
        for i in range(0, actual_examples, batch_size):
            end_index = min(i + batch_size, actual_examples)
            batch_data = dataset[i:end_index]
            
            # Handle both single examples and batches
            if isinstance(batch_data['article'], str):
                batch_articles = [batch_data['article']]
                batch_references = [batch_data['highlights']]
            else:
                batch_articles = batch_data['article']
                batch_references = batch_data['highlights']
            
            references.extend(batch_references)
            
            # Create prompts
            prompts = [f"[INST] Summarize this article:\n\n{article}\n\nSummary: [/INST]" 
                      for article in batch_articles]
            
            # Tokenize and generate
            inputs = tokenizer(
                prompts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=1800  # Leave room for generation
            ).to(model.device)
            
            outputs = model.generate(
                **inputs, 
                max_new_tokens=100, 
                temperature=0.7, 
                do_sample=True, 
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
            # Decode summaries
            batch_summaries = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            cleaned_summaries = [s.split("[/INST]")[-1].strip() for s in batch_summaries]
            summaries.extend(cleaned_summaries)
            
            if (i + batch_size) % 50 == 0 or end_index == actual_examples:
                print(f"  Processed {len(summaries)}/{actual_examples} examples...")

    # Calculate metrics
    print("  Computing ROUGE-L and BERTScore...")
    
    # ROUGE-L
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_scores = [scorer.score(ref, summ)['rougeL'].fmeasure 
                   for ref, summ in zip(references, summaries)]
    rouge_l = np.mean(rouge_scores)
    
    # BERTScore F1
    try:
        _, _, F1 = bert_score(summaries, references, lang="en", verbose=False)
        bert_f1 = F1.mean().item()
    except Exception as e:
        print(f"  BERTScore error: {e}")
        bert_f1 = 0.0
    
    # Length metrics
    gen_lengths = [len(s.split()) for s in summaries]
    ref_lengths = [len(r.split()) for r in references]
    
    return {
        'model': model_name,
        'rouge_l': rouge_l,
        'bert_f1': bert_f1,
        'avg_gen_len': np.mean(gen_lengths),
        'avg_ref_len': np.mean(ref_lengths),
        'num_examples': len(summaries)
    }

def main():
    # Load config
    with open('configs/param.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load models
    base_model, adapter_model, tokenizer = load_models(config)

    # Load test data
    dataset_config = config['dataset']
    test_split = dataset_config.get('test_split', 'test[:1000]')
    print(f"Loading test data: {test_split}")
    
    dataset = load_dataset(
        dataset_config['name'], 
        dataset_config['version'], 
        split=test_split
    )
    
    print("="*60)
    print("BASELINE EVALUATION - ROUGE-L & BERTScore")
    print("="*60)
    
    # Evaluate base model
    num_samples = len(dataset)
    
    # Evaluate adapter if exists
    base_results = evaluate_model(base_model, tokenizer, dataset, "Base Model", num_samples)
    adapter_results = None
    if adapter_model is not None:
        adapter_results = evaluate_model(adapter_model, tokenizer, dataset, "LoRA Adapter", num_samples)

    # Print results
    if base_results:
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        print(f"📊 Base Model ({base_results['num_examples']} examples):")
        print(f"  ROUGE-L: {base_results['rouge_l']:.4f}")
        print(f"  BERTScore F1: {base_results['bert_f1']:.4f}")
        print(f"  Avg Length: {base_results['avg_gen_len']:.1f} words (ref: {base_results['avg_ref_len']:.1f})")
    
    if adapter_results:
        print(f"\n🎯 LoRA Adapter ({adapter_results['num_examples']} examples):")
        print(f"  ROUGE-L: {adapter_results['rouge_l']:.4f}")
        print(f"  BERTScore F1: {adapter_results['bert_f1']:.4f}")
        print(f"  Avg Length: {adapter_results['avg_gen_len']:.1f} words (ref: {adapter_results['avg_ref_len']:.1f})")
        
        # Show improvement
        rouge_diff = adapter_results['rouge_l'] - base_results['rouge_l']
        bert_diff = adapter_results['bert_f1'] - base_results['bert_f1']
        
        print(f"\n📈 Improvement:")
        print(f"  ROUGE-L: {rouge_diff:+.4f}")
        print(f"  BERTScore F1: {bert_diff:+.4f}")
        
        if rouge_diff > 0.02:
            print("\n✅ Adapter is helping significantly!")
            print("   Ready to add hierarchical processing")
        elif rouge_diff > 0:
            print("\n🟡 Adapter shows modest improvement")
            print("   Consider more training or better data")
        else:
            print("\n🔴 Adapter not improving performance")
            print("   Check training setup")
    else:
        print(f"\n⚠️  No adapter found")
        print(f"   Train first with: python -m src.training.train")
    
    print("\n" + "="*60)
    print("🎯 TARGETS:")
    print("  ROUGE-L: >0.35 (excellent), >0.30 (good)")  
    print("  BERTScore F1: >0.88 (excellent), >0.85 (good)")
    print("="*60)

if __name__ == "__main__":
    main()