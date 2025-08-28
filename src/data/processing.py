import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from dataclasses import dataclass
from typing import Any, Dict, List

def _preprocess_function(examples: Dict[str, List[str]], prompt_template: str) -> Dict[str, List[str]]:
    """
    Converts articles and highlights into the instruction format using a template.
    This is a helper function.
    """
    instructions = []
    summaries = []

    for article, highlights in zip(examples['article'], examples['highlights']):
        instruction = prompt_template.format(article=article)
        summary = highlights.strip()
        instructions.append(instruction)
        summaries.append(summary)

    return {'instruction': instructions, 'summary': summaries}

def _tokenize_function(examples: Dict[str, List[str]], tokenizer: AutoTokenizer) -> Dict[str, List]:
    """
    Tokenizes the instruction and summary separately, applies prompt masking.
    """
    input_ids_list = []
    labels_list = []

    for idx, (inst, summ) in enumerate(zip(examples['instruction'], examples['summary'])):
        inst_tokens = tokenizer(inst, add_special_tokens=False, truncation=False)['input_ids']
        
        summ_text = " " + summ + tokenizer.eos_token
        summ_tokens = tokenizer(summ_text, add_special_tokens=False, truncation=False)['input_ids']
        
        full_tokens = inst_tokens + summ_tokens
        
        if len(full_tokens) > tokenizer.model_max_length:
            excess = len(full_tokens) - tokenizer.model_max_length
            inst_text = inst  

            # Find article start and end markers in your template
            article_start_marker = "Summarize the following news article:\n\n"
            article_end_marker = "\n\nSummary:"
            
            start_idx = inst_text.find(article_start_marker) + len(article_start_marker)
            end_idx = inst_text.find(article_end_marker)
            
            if start_idx != -1 and end_idx != -1:
                # Extract parts
                prefix = inst_text[:start_idx]  # "[INST] Summarize the following news article:\n\n"
                article = inst_text[start_idx:end_idx]  # The actual article content
                suffix = inst_text[end_idx:]  # "\n\nSummary: [/INST]"
                
                # Truncate only the article part
                article_tokens = tokenizer(article, add_special_tokens=False)['input_ids']
                if len(article_tokens) > excess:
                    truncated_article_tokens = article_tokens[:-excess]  # Remove from end of article
                    truncated_article = tokenizer.decode(truncated_article_tokens, skip_special_tokens=True)
                    
                    # Reconstruct with truncated article
                    truncated_inst = prefix + truncated_article + suffix
                    inst_tokens = tokenizer(truncated_inst, add_special_tokens=False)['input_ids']
                    
                else:
                    # If article is shorter than excess, use original 
                    inst_tokens = inst_tokens[:-excess]
            else:
                print(f"Could not find article boundaries, using fallback truncation")
                # Fallback to original logic if parsing fails
                inst_tokens = inst_tokens[:-excess]

            full_tokens = inst_tokens + summ_tokens
        
        inst_len = len(inst_tokens)
        # Masking the instruction part for better training
        labels = [-100] * inst_len + summ_tokens

        input_ids_list.append(full_tokens)
        labels_list.append(labels)

    return {
        "input_ids": input_ids_list,
        "labels": labels_list
    }

def load_and_prepare_dataset(dataset_name: str, dataset_version: str, split: str, tokenizer: AutoTokenizer, max_length: int, prompt_template: str):
    """
    Loads and processes a specific split of the dataset.
    """
    print(f"🔄 Loading and preparing split: {split}...")
    
    tokenizer.model_max_length = max_length
    dataset = load_dataset(dataset_name, dataset_version, split=split)
    
    formatted_dataset = dataset.map(
        lambda x: _preprocess_function(x, prompt_template=prompt_template),
        batched=True,
        batch_size=100,
        remove_columns=dataset.column_names
    )
    
    tokenized_dataset = formatted_dataset.map(
        lambda x: _tokenize_function(x, tokenizer),
        batched=True,
        batch_size=100,
        remove_columns=['instruction', 'summary']
    )

    print(f"✅ Split '{split}' ready with {len(tokenized_dataset)} examples.")
    return tokenized_dataset

@dataclass
class CustomDataCollator:
    """
    A custom data collator to handle padding for causal language modeling.
    Simplified version since truncation is handled in preprocessing.
    """
    tokenizer: AutoTokenizer
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        if not features:
            raise ValueError("Empty batch")
        
        input_ids = [torch.tensor(f['input_ids'], dtype=torch.long) for f in features]
        labels = [torch.tensor(f['labels'], dtype=torch.long) for f in features]
        
        # Find max length in current batch
        max_len = max(len(seq) for seq in input_ids)
        
        
        padded_input_ids = []
        padded_labels = []
        attention_masks = []
        
        for idx, (inp, lab) in enumerate(zip(input_ids, labels)):
            pad_len = max_len - len(inp)
            
            # Pad input_ids with pad_token_id
            padded_inp = torch.cat([inp, torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=torch.long)])
            padded_input_ids.append(padded_inp)
            
            # Pad labels with -100 (ignore index)
            padded_lab = torch.cat([lab, torch.full((pad_len,), -100, dtype=torch.long)])
            padded_labels.append(padded_lab)
            
            # Create attention mask (1 for real tokens, 0 for padding)
            attention_mask = torch.cat([torch.ones(len(inp), dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)])
            attention_masks.append(attention_mask)

        batch_labels = torch.stack(padded_labels)
        batch_input_ids = torch.stack(padded_input_ids)
        batch_attention_masks = torch.stack(attention_masks)
        
        # Safety check for trainable tokens
        if (batch_labels != -100).sum().item() == 0:
            print(f"CRITICAL: Batch has no trainable tokens! This will cause NaN gradients.")
            batch_labels[0, 0] = padded_input_ids[0][0].item()
            print(f"Emergency fix applied: made 1 token trainable")
            
        return {
            'input_ids': batch_input_ids,
            'attention_mask': batch_attention_masks,
            'labels': batch_labels
        }