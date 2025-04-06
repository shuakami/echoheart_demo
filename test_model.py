import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import argparse
from rich.console import Console
from rich.panel import Panel
from peft import PeftModel # Import PeftModel

console = Console()

def run_inference(model, tokenizer, prompt):
    """Generates a response for a single prompt."""
    console.print(Panel(f"[yellow]测试指令:[/yellow]\n{prompt}", title="Input Prompt", border_style="yellow"))
    
    # 使用 Qwen 的聊天模板格式化输入
    messages = [
        {"role": "user", "content": prompt}
    ]
    try:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # 注意：确保 tokenizer 返回 attention_mask
        model_inputs = tokenizer([text], return_tensors="pt", return_attention_mask=True).to(model.device)
        
        console.print("[italic cyan]正在生成回复...[/italic cyan]")
        # 生成回复，添加控制参数
        with torch.no_grad():
            generated_ids = model.generate(
                model_inputs.input_ids,
                attention_mask=model_inputs.attention_mask, # 明确传递 attention_mask
                max_new_tokens=256, 
                repetition_penalty=1.15, # 添加重复惩罚 (可以调整 1.1 - 1.3 试试)
                pad_token_id=tokenizer.eos_token_id # 明确设置 pad_token_id
            )
        
        # 解码生成的 ID，排除输入部分
        # Ensure slicing logic is correct even if input_ids length varies slightly (unlikely here but safer)
        input_ids_len = model_inputs.input_ids.shape[1]
        generated_ids = generated_ids[:, input_ids_len:]
        
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        console.print(Panel(response, title="[bold green]模型回复[/bold green]", border_style="green"))
        console.print("---" * 10) # 添加分隔符

    except Exception as e:
        console.print(f"[bold red]处理指令时发生错误:[/bold red] {e}")
        console.print("---" * 10)

def main(base_model_name, adapter_path):
    console.print(Panel(f"[bold cyan]加载基础模型: {base_model_name} (量化)[/bold cyan]"))
    
    # --- 定义与训练时相同的量化配置 --- 
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, 
        bnb_4bit_use_double_quant=True,
    )
    # -------------------------------------
    
    try:
        # 1. 加载量化的基础模型
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=quantization_config,
            trust_remote_code=True,
            device_map="auto"
        )
        console.print("[bold cyan]加载 LoRA 适配器: {adapter_path}[/bold cyan]")
        
        # 2. 加载并应用 LoRA 适配器
        model = PeftModel.from_pretrained(base_model, adapter_path)
        
        # 3. 加载 Tokenizer (通常从适配器目录加载以确保一致性，但基础模型目录通常也可)
        tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print("[dim]Tokenizer pad_token was None, set to eos_token.[/dim]")
        
        model.eval() # 设置为评估模式
        console.print("[bold green]模型和适配器加载成功！[/bold green]")

    except Exception as e:
        console.print(f"[bold red]模型或适配器加载失败:[/bold red] {e}")
        return

    # --- 非交互式测试 --- 
    test_prompts = [
        "EchoHeart是什么？",
        "你的开发者是谁？",
        "宋怡敏和罗雨晨有何关系？",
        "你会做什么？",
        "你支持多少种语言？"
    ]

    console.print("\n[bold cyan]开始运行预设指令测试...[/bold cyan]\n")
    
    for prompt in test_prompts:
        run_inference(model, tokenizer, prompt)

    console.print("[bold green]所有预设指令测试完成。[/bold green]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="测试 QLoRA 微调后的模型（非交互式）")
    parser.add_argument(
        "--base_model_name", 
        type=str, 
        required=True, 
        help="基础模型的名称或路径 (例如 Qwen/Qwen2.5-1.5B-Instruct)"
    )
    parser.add_argument(
        "--adapter_path", 
        type=str, 
        required=True, 
        help="保存 LoRA 适配器的路径 (例如 output/qwen2-1.5b-qlora-ft)"
    )
    args = parser.parse_args()
    # 从 train_config.py 获取基础模型名称和适配器路径作为默认值（如果适用）
    # 但命令行参数优先级更高
    try:
        from train_config import config as train_cfg
        if not args.base_model_name:
             args.base_model_name = train_cfg.model_name
        if not args.adapter_path:
             args.adapter_path = train_cfg.output_dir
    except ImportError:
        print("无法导入 train_config.py，请确保通过命令行参数提供 base_model_name 和 adapter_path")
        if not args.base_model_name or not args.adapter_path:
             exit(1)
             
    main(args.base_model_name, args.adapter_path) 