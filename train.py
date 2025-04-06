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
    tokenizer.pad_token = tokenizer.eos_token
    
    def preprocess_function(examples):
        # 将消息转换为模型输入格式
        conversations = []
        for msg in examples['messages']:
            user_msg = next(m for m in msg if m['role'] == 'user')['content']
            assistant_msg = next(m for m in msg if m['role'] == 'assistant')['content']
            conversation = f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n{assistant_msg}<|im_end|>"
            conversations.append(conversation)
        
        # tokenize
        tokenized = tokenizer(
            conversations,
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt"
        )
        return tokenized

    # 处理数据集
    console.print("[bold cyan]处理数据集...[/bold cyan]")
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset['train'].column_names
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
        seed=config.seed,
        report_to="none",
    )
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
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