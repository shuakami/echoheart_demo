import argparse
from llama_cpp import Llama, LlamaGrammar
from rich.console import Console
from rich.panel import Panel
import sys
import os

console = Console()

def run_gguf_inference(llm: Llama, prompt: str):
    """使用加载的 GGUF 模型运行单轮推理。"""
    console.print(Panel(f"[yellow]测试指令:[/yellow]\n{prompt}", title="Input Prompt", border_style="yellow"))

    # 注意：聊天模板应该在加载 Llama 对象时通过 chat_format 参数设置
    # 这里我们直接构建 messages 列表
    messages = [
        {"role": "system", "content": "You are a helpful assistant."}, # 可以根据需要调整系统提示
        {"role": "user", "content": prompt}
    ]

    try:
        console.print("[italic cyan]正在生成回复...[/italic cyan]")
        
        completion = llm.create_chat_completion(
            messages=messages,
            max_tokens=256,       # 最大生成 token 数
            temperature=0.7,      # 温度，控制随机性
            top_p=0.9,            # Top-p 采样
            repeat_penalty=1.1,   # 重复惩罚
            stop=["<|im_end|>", "<|endoftext|>"] # Qwen 的停止标记
        )

        response_text = completion['choices'][0]['message']['content']
        
        # 尝试去除可能残留的 <|im_start|> assistant

        response_text = response_text.strip()
        # if response_text.startswith("<|im_start|>assistant\n"):
        #     response_text = response_text[len("<|im_start|>assistant\n"):].strip()

        console.print(Panel(response_text, title="[bold green]模型回复 (GGUF)[/bold green]", border_style="green"))
        console.print("--- " * 10)

    except Exception as e:
        console.print(f"[bold red]处理 GGUF 指令时发生错误:[/bold red] {e}")
        import traceback
        traceback.print_exc()
        console.print("--- " * 10)


def main(gguf_model_path: str, n_gpu_layers: int = -1, chat_format: str = "qwen"):
    console.print(Panel(f"[bold cyan]加载 GGUF 模型: {gguf_model_path}[/bold cyan]"))

    if not os.path.exists(gguf_model_path):
        console.print(f"[bold red]错误: GGUF 模型文件不存在: {gguf_model_path}[/bold red]")
        sys.exit(1)

    try:
        llm = Llama(
            model_path=gguf_model_path,
            n_gpu_layers=n_gpu_layers,  # -1 表示尽可能多地使用 GPU 层
            n_ctx=2048,              # 上下文窗口大小，根据模型调整
            chat_format=chat_format,   # 设置聊天格式，对于 Qwen 很重要
            verbose=False             # 可以设为 True 获取更详细的加载信息
        )
        console.print("[bold green]GGUF 模型加载成功！[/bold green]")
    except Exception as e:
        console.print(f"[bold red]加载 GGUF 模型失败:[/bold red] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # --- 非交互式测试 ---
    test_prompts = [
        "EchoHeart是什么？",
        "你的开发者是谁？",
        "宋怡敏和罗雨晨有何关系？",
        "你会做什么？",
        "用 Python 写一个快速排序算法。"
    ]

    console.print("\n[bold cyan]开始运行预设指令测试 (GGUF)...[/bold cyan]\n")

    for prompt in test_prompts:
        run_gguf_inference(llm, prompt)

    console.print("[bold green]所有预设指令测试完成 (GGUF)。[/bold green]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="测试 GGUF 格式的模型（非交互式）")
    parser.add_argument(
        "--gguf_model_path",
        type=str,
        required=True,
        help="GGUF 模型文件的路径"
    )
    parser.add_argument(
        "--n_gpu_layers",
        type=int,
        default=-1,
        help="卸载到 GPU 的层数 (-1 表示全部可能)"
    )
    parser.add_argument(
        "--chat_format",
        type=str,
        default="qwen", # 默认为 qwen 格式
        help="llama-cpp-python 支持的聊天格式 (例如 'llama-2', 'chatml', 'qwen'等)"
    )
    args = parser.parse_args()
    main(args.gguf_model_path, args.n_gpu_layers, args.chat_format) 