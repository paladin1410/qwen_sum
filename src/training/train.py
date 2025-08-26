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
    """
    A custom callback to log metrics from the Trainer to the main logger.
    """
    def on_log(self, args, state, control, logs=None, **kwargs):
        # Called by the Trainer to log loss metrics.
        if state.is_world_process_zero: # Only log on the main process
            if logs:
                # Format the logs into a readable string
                log_str = " ".join(f"{key}: {value}" for key, value in logs.items())
                logging.info(f"TRAINER_LOGS - {log_str}")

def main():
    """Main function to run the training pipeline."""

    # Load Configuration
    with open('configs/param.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    training_config = config['training']
    output_dir = training_config['output_dir']

    # Ensure Output Directory Exists ---
    os.makedirs(output_dir, exist_ok=True)
    
    # Configure Logging 
    log_file_path = os.path.join(output_dir, "training.log")
    
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(log_file_path),  
            logging.StreamHandler(sys.stdout),
        ]
    )
    
    logging.info(f"📝 Logging to file: {log_file_path}")
    logging.info("🔧 Configuration loaded successfully.")
    
    model_config = config['model']
    lora_config_params = config['lora']
    
    # Load Tokenizer and Model
    logging.info(f"🚀 Loading base model and tokenizer from: {model_config['base_path']}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_config['base_path'], 
        local_files_only=True
    )
    # Set pad token if it's not already set
    if tokenizer.pad_token is None:
        logging.info("Tokenizer pad_token not set. Setting it to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.truncation_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_config['base_path'],
        torch_dtype=torch.float16,
        trust_remote_code=True,
        local_files_only=True,
    )
    tokenizer.model_max_length = model_config['max_length']

    # Configure and Apply LoRA
    logging.info("🛠️  Configuring LoRA adapter...")
    lora_config = LoraConfig(
        r=lora_config_params['r'],
        lora_alpha=lora_config_params['alpha'],
        target_modules=lora_config_params['target_modules'],
        lora_dropout=lora_config_params['dropout'],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    logging.info("✅ LoRA adapter applied successfully!")
    model.print_trainable_parameters()
    
    # Load and Prepare Datasets
    logging.info("🔄 Loading and preparing train/validation datasets...")
    dataset_config = config['dataset']

    # Load the training dataset 
    train_dataset = load_and_prepare_dataset(
        dataset_name=dataset_config['name'],
        dataset_version=dataset_config['version'],
        split=dataset_config['train_split'],
        tokenizer=tokenizer,
        max_length=config['model']['max_length']
    )

    # Load the validation dataset 
    eval_dataset = load_and_prepare_dataset(
        dataset_name=dataset_config['name'],
        dataset_version=dataset_config['version'],
        split=dataset_config['validation_split'],
        tokenizer=tokenizer,
        max_length=config['model']['max_length']
    )

    # Set up Trainer
    logging.info("🏋️ Setting up training arguments and trainer...")
    training_args = TrainingArguments(
        output_dir=output_dir, 
        num_train_epochs=training_config['num_epochs'],
        per_device_train_batch_size=training_config['batch_size'],
        gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
        learning_rate=training_config['learning_rate'],
        warmup_steps=training_config['warmup_steps'],
        logging_steps=training_config['logging_steps'],
        
        # Updated checkpointing and evaluation strategy
        save_strategy=training_config.get('save_strategy', 'steps'),
        save_steps=training_config['save_steps'],
        eval_strategy=training_config.get('eval_strategy', 'steps'),
        eval_steps=training_config['eval_steps'],
        
        # Load best model parameters from yaml
        metric_for_best_model=training_config.get('metric_for_best_model'),
        load_best_model_at_end=training_config.get('load_best_model_at_end', False),
        greater_is_better=training_config.get('greater_is_better', False),
        save_total_limit=training_config.get('save_total_limit', 10),

        fp16=training_config['fp16'],
        report_to="none", 
        dataloader_num_workers=0,
        optim="paged_adamw_8bit",
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
    )

    data_collator = CustomDataCollator(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[LogCallback()], 
    )

    # Start Training
    logging.info("🚀 Starting training...")
    trainer.train()
    logging.info("🎉 Training completed successfully!")
    
    # Save the Final Adapter
    adapter_path = training_config['adapter_save_path']
    logging.info(f"💾 Saving final LoRA adapter to {adapter_path}...")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    logging.info(f"✅ Adapter saved to {adapter_path}.")

if __name__ == "__main__":
    main()