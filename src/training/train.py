import sys
import os
import yaml
import torch
import logging
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, TaskType
from src.data.processing import load_and_prepare_dataset, CustomDataCollator

class LogCallback(TrainerCallback):
    """A custom callback to log metrics to the main logger."""
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero and logs:
            log_str = " ".join(f"{key}: {value}" for key, value in logs.items())
            logging.info(f"TRAINER_LOGS - {log_str}")

def main():
    """Main function to run the training pipeline."""
    print(f"Number of GPUs detected: {torch.cuda.device_count()}")

    with open('configs/param.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    training_config = config['training']
    model_config = config['model']
    dataset_config = config['dataset']
    lora_config_params = config['lora']
    prompt_template = config['prompt']['template']

    output_dir = training_config['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    log_file_path = os.path.join(output_dir, "training.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file_path), logging.StreamHandler(sys.stdout)]
    )
    
    logging.info("Configuration loaded successfully.")
    
    tokenizer = AutoTokenizer.from_pretrained(model_config['base_path'], local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_config['base_path'],
        torch_dtype=torch.float16,
        trust_remote_code=True,
        local_files_only=True,
    )
    tokenizer.model_max_length = model_config['max_length']

    lora_config = LoraConfig(
        r=lora_config_params['r'],
        lora_alpha=lora_config_params['alpha'],
        target_modules=lora_config_params['target_modules'],
        lora_dropout=lora_config_params['dropout'],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    train_dataset = load_and_prepare_dataset(
        dataset_name=dataset_config['name'],
        dataset_version=dataset_config['version'],
        split=dataset_config['train_split'],
        tokenizer=tokenizer,
        max_length=config['model']['max_length'],
        prompt_template=prompt_template
    )
    eval_dataset = load_and_prepare_dataset(
        dataset_name=dataset_config['name'],
        dataset_version=dataset_config['version'],
        split=dataset_config['validation_split'],
        tokenizer=tokenizer,
        max_length=config['model']['max_length'],
        prompt_template=prompt_template
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=training_config['num_epochs'],
        per_device_train_batch_size=training_config['batch_size'],
        gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
        learning_rate=training_config['learning_rate'],
        warmup_steps=training_config['warmup_steps'],
        logging_steps=training_config['logging_steps'],
        save_strategy='steps',
        save_steps=training_config['save_steps'],
        eval_strategy='steps',
        eval_steps=training_config['eval_steps'],
        metric_for_best_model=training_config['metric_for_best_model'],
        load_best_model_at_end=training_config['load_best_model_at_end'],
        greater_is_better=training_config['greater_is_better'],
        save_total_limit=training_config['save_total_limit'],
        fp16=training_config['fp16'],
        optim=training_config['optimizer'],
        max_grad_norm=training_config['max_grad_norm'],
        lr_scheduler_type=training_config['lr_scheduler_type'],
        report_to=training_config['report_to'],
        dataloader_num_workers=0,
    )

    logging.info(f"EFFECTIVE TrainingArguments: {training_args}")

    data_collator = CustomDataCollator(
        tokenizer=tokenizer,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[LogCallback()],
    )

    logging.info("🚀 Starting training...")

    trainer.train()
    
    adapter_path = training_config['adapter_save_path']
    logging.info(f"Saving final LoRA adapter to {adapter_path}...")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    logging.info(f"Adapter saved to {adapter_path}.")

if __name__ == "__main__":
    main()