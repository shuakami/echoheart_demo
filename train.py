import os
import json
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
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
    # --- QLoRA 配置 --- 
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, # 使用 bfloat16 进行计算
        bnb_4bit_use_double_quant=True, # 使用双量化
    )
    # ---------------------
    
    # 准备数据集
    dataset, tokenizer = prepare_dataset(config.train_dataset_path)
    
    # 加载模型 (使用量化配置)
    console.print("[bold cyan]加载量化模型 (QLoRA)...[/bold cyan]")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=quantization_config, # 应用量化配置
        trust_remote_code=True,
        device_map="auto" # 自动映射设备
    )
    
    # --- PEFT 配置 --- 
    model = prepare_model_for_kbit_training(model) # 准备模型进行 k-bit 训练
    
    # 查找所有线性层以应用 LoRA
    # 对于 Qwen2，通常是 q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
    # 可以通过打印 model 来确认具体层名
    # print(model)
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    lora_config = LoraConfig(
        r=16, # LoRA 秩 (rank)，可以尝试 8, 16, 32, 64
        lora_alpha=32, # LoRA alpha，通常是 r 的两倍
        target_modules=target_modules,
        lora_dropout=0.05, # Dropout 比例
        bias="none", # 通常设置为 none
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config) # 应用 PEFT 配置
    console.print("[bold green]PEFT LoRA 配置已应用。[/bold green]")
    model.print_trainable_parameters() # 打印可训练参数量
    # -------------------
    
    # --- 确定混合精度设置 --- 
    use_bf16 = torch.cuda.is_bf16_supported()
    use_fp16 = not use_bf16 and config.fp16 # 仅在不支持 bf16 时才使用 fp16
    console.print(f"[dim]混合精度设置: BF16={'可用' if use_bf16 else '不可用'}, FP16={'启用' if use_fp16 else '禁用'}[/dim]")
    # --------------------------
    
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
        fp16=use_fp16, # <--- 使用计算后的值
        bf16=use_bf16, # <--- 使用计算后的值
        gradient_checkpointing=True, # 仍然启用梯度检查点
        optim="paged_adamw_8bit", # 使用 bitsandbytes 提供的优化器以节省显存
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
    
    # 保存适配器模型 (LoRA 参数)
    console.print("[bold cyan]保存 LoRA 适配器模型...[/bold cyan]")
    # Trainer 会自动处理 PEFT 模型的保存
    trainer.save_model() 
    # tokenizer 通常也需要保存，以防有更改
    tokenizer.save_pretrained(config.output_dir)
    
    console.print("QLoRA 训练完成！适配器保存在:", config.output_dir)

if __name__ == "__main__":
    main()