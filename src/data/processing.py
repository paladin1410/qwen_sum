import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from dataclasses import dataclass
from typing import Any, Dict, List

def _preprocess_function(examples: Dict[str, List[str]], tokenizer: AutoTokenizer) -> Dict[str, List]:
    """
    Converts articles and highlights into the instruction format for the model.
    This is a helper function.
    """
    instructions = []
    summaries = []

    for article, highlights in zip(examples['article'], examples['highlights']):
        instruction = f"[INST] Summarize the following news article:\n\n{article}\n\nSummary: [/INST]"
        summary = highlights.strip()
        instructions.append(instruction)
        summaries.append(summary)

    return {'instruction': instructions, 'summary': summaries}

# def _tokenize_function(examples: Dict[str, List[str]], tokenizer: AutoTokenizer) -> Dict[str, List]:
#     """
#     Tokenizes the combined instruction and summary for training.
#     This is a helper function.
#     """
#     full_texts = [inst + " " + summ + tokenizer.eos_token for inst, summ in zip(examples['instruction'], examples['summary'])]
    
#     tokenized = tokenizer(
#         full_texts,
#         truncation=True,
#         max_length=tokenizer.model_max_length,
#         padding=False,
#         return_tensors=None,
#     )
#     # For causal LM, labels are the same as input_ids
#     tokenized["labels"] = tokenized["input_ids"].copy()
#     return tokenized

def _tokenize_function(examples: Dict[str, List[str]], tokenizer: AutoTokenizer) -> Dict[str, List]:
    """
    Tokenizes the instruction and summary separately, applies prompt masking.
    """
    input_ids_list = []
    labels_list = []

    for inst, summ in zip(examples['instruction'], examples['summary']):
        # Tokenize instruction (prompt + article) without adding BOS/EOS yet
        inst_tokens = tokenizer(inst, add_special_tokens=False, truncation=False)['input_ids']
        
        # Tokenize summary with leading space and EOS
        summ_text = " " + summ + tokenizer.eos_token
        summ_tokens = tokenizer(summ_text, add_special_tokens=False, truncation=False)['input_ids']
        
        # Concatenate: instruction + summary
        full_tokens = inst_tokens + summ_tokens
        
        # Truncate if exceeding max_length (prioritize summary by truncating instruction from end)
        if len(full_tokens) > tokenizer.model_max_length:
            excess = len(full_tokens) - tokenizer.model_max_length
            inst_tokens = inst_tokens[:-excess]  # Truncate instruction
            full_tokens = inst_tokens + summ_tokens
        
        # Labels: -100 for instruction, copy for summary
        inst_len = len(inst_tokens)
        # summ_tokens already includes space + summ + EOS
        labels = [-100] * inst_len + summ_tokens  
        
        input_ids_list.append(full_tokens)
        labels_list.append(labels)

    return {
        "input_ids": input_ids_list,
        "labels": labels_list
    }


def load_and_prepare_dataset(dataset_name, dataset_version, split, tokenizer, max_length):
    """
    Loads and processes a specific split of the dataset.
    
    Args:
        dataset_name (str): The name of the dataset from Hugging Face.
        dataset_version (str): The version of the dataset.
        split (str): The specific split to load (e.g., "train[:50000]").
        tokenizer (AutoTokenizer): The tokenizer to use.
        max_length (int): The maximum sequence length.
        
    Returns:
        The processed dataset for the given split.
    """
    print(f"🔄 Loading and preparing split: {split}...")
    
    # Set the max_length on the tokenizer so the helper function can access it
    tokenizer.model_max_length = max_length

    # Load the specified split from Hugging Face Hub
    dataset = load_dataset(dataset_name, dataset_version, split=split)
    
    # 1. Apply instruction formatting 
    formatted_dataset = dataset.map(
        lambda x: _preprocess_function(x, tokenizer),
        batched=True,
        batch_size=100,
        remove_columns=dataset.column_names
    )
    
    # 2. Apply tokenization 
    tokenized_dataset = formatted_dataset.map(
        lambda x: _tokenize_function(x, tokenizer),
        batched=True,
        batch_size=100,
        remove_columns=['instruction', 'summary']
    )

    # # Print to debug a sample after preprocessing
    # if len(tokenized_dataset) > 0:
    #     print("Sample input_ids:", tokenized_dataset[0]["input_ids"])
    #     print("Sample labels:", tokenized_dataset[0]["labels"])
    #     # Optional: Decode to text for readability
    #     print("Decoded input:", tokenizer.decode(tokenized_dataset[0]["input_ids"]))
    #     print("Decoded labels (ignoring -100):", tokenizer.decode([t for t in tokenized_dataset[0]["labels"] if t != -100]))
    

    print(f"✅ Split '{split}' ready with {len(tokenized_dataset)} examples.")
    
    # Return the single, processed dataset
    return tokenized_dataset

@dataclass
class CustomDataCollator:
    """
    A custom data collator to handle padding for causal language modeling.
    This ensures that padding tokens are ignored in the loss calculation.
    """
    tokenizer: AutoTokenizer
    mlm: bool = False

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Basic validation
        if not features:
            raise ValueError("Empty batch")
        
        input_ids = [torch.tensor(f['input_ids'], dtype=torch.long) for f in features]
        labels = [torch.tensor(f['labels'], dtype=torch.long) for f in features]
        
        max_len = max(len(seq) for seq in input_ids)
        
        # Safety cap to prevent OOM
        if max_len > 4096:
            max_len = 4096
            print(f"⚠️  Capping batch max_len to {max_len}")
        
        padded_input_ids = []
        padded_labels = []
        attention_masks = []

        for inp, lab in zip(input_ids, labels):
            # Truncate if needed
            if len(inp) > max_len:
                # Find last trainable token (not -100) to preserve summary
                trainable_positions = [i for i, label in enumerate(lab) if label != -100]
                if trainable_positions:
                    last_trainable = trainable_positions[-1]
                    # Keep from start up to last trainable + small buffer, within max_len
                    keep_until = min(len(inp), max(last_trainable + 5, max_len))
                    if keep_until > max_len:
                        # If we must truncate, cut from beginning but keep end
                        cut_amount = len(inp) - max_len
                        inp = inp[cut_amount:]
                        lab = lab[cut_amount:]
                    else:
                        inp = inp[:keep_until]
                        lab = lab[:keep_until]
                else:
                    # Fallback: normal truncation but this shouldn't happen
                    inp = inp[:max_len]
                    lab = lab[:max_len]
            
            pad_len = max_len - len(inp)
            
            # Pad input_ids with the tokenizer's pad_token_id
            padded_inp = torch.cat([inp, torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=torch.long)])
            padded_input_ids.append(padded_inp)

            # Pad labels with -100 to be ignored by the loss function
            padded_lab = torch.cat([lab, torch.full((pad_len,), -100, dtype=torch.long)])
            padded_labels.append(padded_lab)

            # Create attention mask
            attention_mask = torch.cat([torch.ones(len(inp), dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)])
            attention_masks.append(attention_mask)

        # CRITICAL: Validate batch has trainable tokens to prevent NaN
        batch_labels = torch.stack(padded_labels)
        total_trainable = (batch_labels != -100).sum().item()
        
        if total_trainable == 0:
            print(f"❌ CRITICAL: Batch has no trainable tokens! This will cause NaN gradients.")
            print(f"Batch size: {len(padded_input_ids)}, Max len: {max_len}")
            # Make first token of first sequence trainable
            batch_labels[0, 0] = padded_input_ids[0][0].item()
            print(f"🚨 Emergency fix applied: made 1 token trainable")
        
        return {
            'input_ids': torch.stack(padded_input_ids),
            'attention_mask': torch.stack(attention_masks),
            'labels': batch_labels
        }