import os
import json
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset
from train_config import config
from logger_utils import CustomTrainerCallback, console

def prepare_dataset(dataset_path):
    # 加载数据集
    console.print("[bold cyan]加载数据集...[/bold cyan]")
    dataset = load_dataset('json', data_files=dataset_path)
    
    # 获取tokenizer
    console.print("[bold cyan]加载tokenizer...[/bold cyan]")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True)
    
    # 确保设置 pad_token, 对于 Qwen 通常设置为 <|endoftext|> 或 eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token 
        # 或者 tokenizer.pad_token = "<|endoftext|>"
        print(f"[dim]Tokenizer pad_token was None, set to {tokenizer.pad_token}.[/dim]")

    # Qwen的模板通常不需要在末尾添加EOS，apply_chat_template会处理
    # 参考: https://huggingface.co/Qwen/Qwen1.5-0.5B-Chat/blob/main/tokenizer_config.json
    # chat_template: "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}{{ '<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n' }}{% endif %}{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    
    def preprocess_function(examples):
        # 使用 tokenizer.apply_chat_template 处理对话格式
        formatted_conversations = []
        for i in range(len(examples['messages'])): # Iterate over samples in the batch
             # Extract messages for the current sample
             current_sample_messages = examples['messages'][i]
             # Ensure it's a list of dicts as expected by apply_chat_template
             if isinstance(current_sample_messages, list): 
                 try:
                     # Apply chat template, DO NOT add generation prompt during training
                     formatted = tokenizer.apply_chat_template(
                         current_sample_messages, 
                         tokenize=False, 
                         add_generation_prompt=False # Important for training!
                     )
                     formatted_conversations.append(formatted)
                 except Exception as e:
                     print(f"Error applying template to sample {i}: {current_sample_messages}. Error: {e}")
                     formatted_conversations.append("") # Append empty string on error
             else:
                 print(f"Skipping invalid sample format at index {i}: {current_sample_messages}")
                 formatted_conversations.append("")

        # Tokenize the formatted conversations
        tokenized = tokenizer(
            formatted_conversations, 
            truncation=True, 
            max_length=512, 
            padding="max_length", # Pad to max_length
            return_tensors="pt"
        )
        
        # For Causal LM, labels are typically the input_ids shifted
        # The trainer usually handles this automatically if labels aren't provided explicitly
        # or if using DataCollatorForLanguageModeling.
        # However, let's make it explicit for clarity, masking padding tokens.
        labels = tokenized["input_ids"].clone()
        # Mask padding token labels
        labels[labels == tokenizer.pad_token_id] = -100 
        tokenized["labels"] = labels
        
        return tokenized

    # 处理数据集
    console.print("[bold cyan]处理数据集...[/bold cyan]")
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True, # Process in batches
        remove_columns=dataset['train'].column_names # Remove original columns
    )
    
    return tokenized_dataset, tokenizer

def main():
    # 准备数据集
    dataset, tokenizer = prepare_dataset(config.train_dataset_path)
    
    # 加载模型
    console.print("[bold cyan]加载模型...[/bold cyan]")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        device_map="auto"
    )
    
    # 设置训练参数
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        save_steps=config.save_steps,
        logging_steps=config.logging_steps,
        fp16=config.fp16,
        gradient_checkpointing=True,
        seed=config.seed,
        report_to="none",
    )
    
    # 创建训练器
    # Use DataCollatorForLanguageModeling, it handles shifting labels for Causal LM
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        data_collator=data_collator, # Use the data collator
        callbacks=[CustomTrainerCallback()], 
    )
    
    # 开始训练
    trainer.train()
    
    # 保存模型
    console.print("[bold cyan]保存模型...[/bold cyan]")
    trainer.save_model()
    tokenizer.save_pretrained(config.output_dir)

if __name__ == "__main__":
    main()