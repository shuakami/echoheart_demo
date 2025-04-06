import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from rich.console import Console
from rich.panel import Panel

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

def main(model_path):
    console.print(Panel(f"[bold cyan]正在加载模型和tokenizer: {model_path}[/bold cyan]"))
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        # 确保 tokenizer 有 pad_token，如果没有，设置为 eos_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print("[dim]Tokenizer pad_token was None, set to eos_token.[/dim]")
            
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map="auto" # 自动选择设备 (CPU/GPU)
        )
        model.eval() # 设置为评估模式
        console.print("[bold green]模型加载成功！[/bold green]")
    except Exception as e:
        console.print(f"[bold red]模型加载失败:[/bold red] {e}")
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
    parser = argparse.ArgumentParser(description="测试微调后的Qwen模型（非交互式）")
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="output/qwen-ft", 
        help="微调后模型的保存路径"
    )
    args = parser.parse_args()
    main(args.model_path) 