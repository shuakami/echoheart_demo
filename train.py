import os
import json
import torch
import argparse
import gc # 引入垃圾回收模块，辅助清理
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader # 用于创建测试 DataLoader
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from logger_utils import CustomTrainerCallback, console

# 导入 ModelScope 相关工具
try:
    import modelscope_utils
    modelscope_available = True
except ImportError:
    modelscope_available = False
    print("注意: modelscope_utils 模块未找到。如果无法连接到 Hugging Face，将无法回退到 ModelScope。")

def get_gpu_memory_gb():
    """获取可用 GPU 的总显存 (GB)，如果没有 GPU 则返回 0。"""
    if torch.cuda.is_available():
        total_mem_bytes = torch.cuda.get_device_properties(0).total_memory
        total_mem_gb = total_mem_bytes / (1024**3)
        return total_mem_gb
    return 0

def find_max_batch_size(model, tokenizer, dataset: Dataset, initial_batch_size=8):
    """
    通过试错法动态查找单个 GPU 能容纳的最大 per_device_train_batch_size。
    """
    console.print(f"[bold blue]开始动态查找最大可用 batch size (初始尝试: {initial_batch_size})...[/bold blue]")
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # 从初始值开始尝试，逐步减小
    test_batch_size = initial_batch_size
    max_found_batch_size = 0

    while test_batch_size >= 1:
        # 确保 test_batch_size 是整数
        current_test_bs = int(test_batch_size)
        if current_test_bs < 1: # 防止变成0或负数
             break 
             
        console.print(f"[dim]尝试 batch_size = {current_test_bs}...[/dim]")
        try:
            # 只取一小批数据进行测试
            if len(dataset) < current_test_bs:
                 console.print(f"[yellow]警告：数据集大小 ({len(dataset)}) 小于测试 batch size ({current_test_bs})。跳过此 batch size。[/yellow]")
                 test_batch_size //= 2 # 直接尝试更小的
                 continue
                 
            small_dataset = dataset.select(range(current_test_bs)) 
            
            # 手动模拟一个训练步骤
            model.train() # 确保模型在训练模式
            
            # 准备批次数据
            raw_batch = [small_dataset[i] for i in range(current_test_bs)]
            batch = data_collator(raw_batch)
            
            # 将数据移到 GPU (如果可用)
            if torch.cuda.is_available():
                 batch = {k: v.to(model.device) for k, v in batch.items()}

            # --- 执行前向和反向传播 ---
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_bf16_supported()): # 模拟混合精度
                outputs = model(**batch)
                loss = outputs.loss
                
            # 反向传播 (这是显存消耗的关键部分)
            loss.backward() 
            # --------------------------

            # 如果成功到达这里，说明这个 batch size 可行
            max_found_batch_size = current_test_bs
            console.print(f"[bold green]成功！找到可用最大 batch_size = {max_found_batch_size}[/bold green]")
            
            # 清理显存为下一次迭代或实际训练做准备
            model.zero_grad(set_to_none=True) # 更推荐 set_to_none=True
            del loss, outputs, batch, raw_batch, small_dataset
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            return max_found_batch_size # 找到即可返回

        except torch.cuda.OutOfMemoryError:
            console.print(f"[yellow]显存不足 (OOM) for batch_size = {current_test_bs}。正在尝试更小的...[/yellow]")
            
            # 清理显存
            model.zero_grad(set_to_none=True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # 减小 batch size，通常减半尝试
            test_batch_size //= 2 
            
        except Exception as e:
             console.print(f"[bold red]测试 batch size {current_test_bs} 时发生意外错误: {e}[/bold red]")
             # 清理显存
             model.zero_grad(set_to_none=True)
             gc.collect()
             if torch.cuda.is_available():
                 torch.cuda.empty_cache()
             test_batch_size //= 2 # 仍然尝试更小的

    if max_found_batch_size == 0:
         console.print("[bold red]错误：即使 batch_size = 1 也无法运行。请检查模型大小、数据或 GPU 状态。[/bold red]")
         # 可以选择退出或返回 1 作为最后的尝试
         return 1 # 或者 raise RuntimeError("无法找到合适的 batch size")
         
    # 确保返回的是整数
    return int(max_found_batch_size)

def prepare_dataset(dataset_path, model_name):
    # 加载数据集
    console.print("[bold cyan]加载数据集...[/bold cyan]")
    dataset = load_dataset('json', data_files=dataset_path)
    
    # 获取tokenizer
    console.print("[bold cyan]加载tokenizer...[/bold cyan]")
    
    # 使用 modelscope_utils 加载 tokenizer (如果可用)
    if modelscope_available:
        try:
            _, tokenizer = modelscope_utils.get_model_from_either(
                model_name,
                None,  # 不需要加载模型
                AutoTokenizer,
                use_modelscope=None,  # 自动决定
                trust_remote_code=True
            )
            if tokenizer is None:
                raise ValueError("tokenizer 加载失败")
        except Exception as e:
            console.print(f"[yellow]从 ModelScope 加载 tokenizer 失败: {e}，回退到常规方法[/yellow]")
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
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

def main(args):
    # --- QLoRA 配置 --- 
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, # 使用 bfloat16 进行计算
        bnb_4bit_use_double_quant=True, # 使用双量化
    )
    # ---------------------
    
    # 准备数据集 (在模型加载前，因为测试 batch size 需要数据集)
    dataset, tokenizer = prepare_dataset(args.dataset_file, args.base_model_name)
    
    # 加载模型 (使用量化配置和命令行参数)
    console.print("[bold cyan]加载量化模型 (QLoRA)...[/bold cyan]")
    
    # 尝试使用 ModelScope (如果可用)
    if modelscope_available:
        try:
            model, _ = modelscope_utils.get_model_from_either(
                args.base_model_name,
                AutoModelForCausalLM,
                None,  # 不需要加载 tokenizer，已经加载了
                use_modelscope=None,  # 自动决定
                quantization_config=quantization_config,
                trust_remote_code=True,
                device_map="auto"
            )
        except Exception as e:
            console.print(f"[yellow]从 ModelScope 加载模型失败: {e}，回退到常规方法[/yellow]")
            model = AutoModelForCausalLM.from_pretrained(
                args.base_model_name,
                quantization_config=quantization_config,
                trust_remote_code=True,
                device_map="auto"
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model_name,
            quantization_config=quantization_config,
            trust_remote_code=True,
            device_map="auto"
        )
    
    # --- PEFT 配置 --- 
    model = prepare_model_for_kbit_training(model) # 准备模型进行 k-bit 训练
    
    # 查找所有线性层以应用 LoRA
    # 对于 Qwen2，通常是 q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
    # 可以通过打印 model 来确认具体层名
    # print(model)
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    lora_config = LoraConfig(
        r=args.lora_r, # LoRA 秩 (rank)，可以尝试 8, 16, 32, 64
        lora_alpha=args.lora_alpha, # LoRA alpha，通常是 r 的两倍
        target_modules=target_modules,
        lora_dropout=args.lora_dropout, # Dropout 比例
        bias="none", # 通常设置为 none
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config) # 应用 PEFT 配置
    console.print("[bold green]PEFT LoRA 配置已应用。[/bold green]")
    model.print_trainable_parameters() # 打印可训练参数量
    # -------------------

    # --- 动态确定 Batch Size 和 Grad Acc ---
    # 根据总显存估算一个合理的初始尝试值
    gpu_mem_gb = get_gpu_memory_gb()
    initial_test_bs = 8
    if gpu_mem_gb > 24:
        initial_test_bs = 16
    elif gpu_mem_gb < 10:
         initial_test_bs = 4
         
    # 运行测试找到最大 batch size
    determined_batch_size = find_max_batch_size(
        model, 
        tokenizer, 
        dataset['train'], # 使用训练集的一小部分
        initial_batch_size=initial_test_bs 
    )
    
    # 计算梯度累积步数
    target_effective_batch_size = 16 # 目标有效批次大小
    gradient_accumulation_steps = max(1, target_effective_batch_size // determined_batch_size)
    console.print(f"[bold magenta]动态确定参数: batch_size={determined_batch_size}, grad_acc={gradient_accumulation_steps}[/bold magenta]")
    # -----------------------------------------
    
    # --- 确定混合精度设置 --- 
    use_bf16 = torch.cuda.is_bf16_supported()
    # 注意：config.fp16 可能不再需要，因为我们会基于 GPU 能力自动选择
    # 但为了保留配置选项，暂时保留，可以考虑未来移除 config 中的 fp16
    use_fp16 = not use_bf16 # 优先 bf16，否则用 fp16
    console.print(f"[dim]混合精度设置: BF16={'可用' if use_bf16 else '不可用'}, FP16={'启用' if use_fp16 else '禁用'}[/dim]")
    # --------------------------
    
    # 设置训练参数 (使用动态确定的值和命令行参数)
    training_args = TrainingArguments(
        output_dir=args.output_dir,             # <--- 使用 args
        num_train_epochs=args.num_train_epochs, # <--- 使用 args
        per_device_train_batch_size=determined_batch_size, 
        gradient_accumulation_steps=gradient_accumulation_steps, 
        learning_rate=args.learning_rate,       # <--- 使用 args
        weight_decay=args.weight_decay,         # <--- 使用 args
        max_grad_norm=args.max_grad_norm,       # <--- 使用 args
        save_steps=args.save_steps,             # <--- 使用 args
        logging_steps=args.logging_steps,         # <--- 使用 args
        fp16=use_fp16, 
        bf16=use_bf16, 
        gradient_checkpointing=True, 
        optim="paged_adamw_8bit", 
        seed=args.seed,                         # <--- 使用 args
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
    trainer.save_model() 
    # tokenizer 通常也需要保存，以防有更改
    tokenizer.save_pretrained(args.output_dir) # <--- 使用 args
    
    console.print("QLoRA 训练完成！适配器保存在:", args.output_dir) # <--- 使用 args

if __name__ == "__main__":
    # --- 添加命令行参数解析 --- 
    parser = argparse.ArgumentParser(description="使用 QLoRA 微调语言模型")
    
    # 模型和数据路径
    parser.add_argument("--base_model_name", type=str, required=True, 
                        help="要微调的基础 Hugging Face 模型名称或路径")
    parser.add_argument("--dataset_file", type=str, required=True, 
                        help="包含 'messages' 字段的 JSON 数据集文件路径")
    parser.add_argument("--output_dir", type=str, default=None, 
                        help="训练输出（适配器）保存目录。如果未指定，将基于模型名称自动生成。")
                        
    # 训练超参数
    parser.add_argument("--num_train_epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="权重衰减")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="梯度裁剪范数")
    
    # LoRA 参数
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA 秩 (rank)")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    # target_modules 通常基于模型架构，暂时硬编码，如果需要也可以添加参数
    
    # 保存、日志和种子
    parser.add_argument("--save_steps", type=int, default=25, help="每 N 步保存一次 checkpoint")
    parser.add_argument("--logging_steps", type=int, default=5, help="每 N 步记录一次日志")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    
    args = parser.parse_args()
    
    # 如果未指定 output_dir，则自动生成
    if args.output_dir is None:
        safe_model_name = args.base_model_name.lower().replace('/', '-')
        args.output_dir = f"output/{safe_model_name}-qlora-ft"
        console.print(f"[dim]未指定输出目录，将使用自动生成的路径: {args.output_dir}[/dim]")
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
        
    # --------------------------
    main(args) # <--- 将解析后的参数传递给 main 函数