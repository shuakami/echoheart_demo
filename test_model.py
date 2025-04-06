import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
from rich.console import Console
from rich.panel import Panel

console = Console()

def main(model_path):
    console.print(Panel(f"[bold cyan]正在加载模型和tokenizer: {model_path}[/bold cyan]"))
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
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

    console.print("\n进入交互模式。输入 'quit' 或 'exit' 退出。")
    
    while True:
        try:
            prompt = console.input("[yellow]请输入您的指令 > [/yellow]")
            if prompt.lower() in ["quit", "exit"]:
                break
            
            if not prompt:
                continue

            # 使用 Qwen 的聊天模板格式化输入
            messages = [
                {"role": "user", "content": prompt}
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
            
            console.print("[italic cyan]正在生成回复...[/italic cyan]")
            # 生成回复
            with torch.no_grad():
                generated_ids = model.generate(
                    model_inputs.input_ids,
                    max_new_tokens=512 # 可根据需要调整最大生成长度
                )
            
            # 解码生成的 ID，排除输入部分
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            console.print(Panel(response, title="[bold green]模型回复[/bold green]", border_style="green"))

        except KeyboardInterrupt:
            break
        except Exception as e:
            console.print(f"[bold red]发生错误:[/bold red] {e}")

    console.print("\n[bold yellow]退出交互模式。[/bold yellow]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="测试微调后的Qwen模型")
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="output/qwen-ft", 
        help="微调后模型的保存路径"
    )
    args = parser.parse_args()
    main(args.model_path) 