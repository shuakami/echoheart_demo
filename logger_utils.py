import time
import psutil
import torch
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel
from transformers import TrainerCallback
import nvidia.ml as nvml

console = Console()

class CustomTrainerCallback(TrainerCallback):
    def __init__(self):
        self.training_start = None
        self.epoch_start = None
        self.progress = Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            TimeElapsedColumn(),
            console=console
        )
        self.task = None
        nvml.nvmlInit()
        self.handle = nvml.nvmlDeviceGetHandleByIndex(0)
        
    def _get_gpu_info(self):
        try:
            info = nvml.nvmlDeviceGetMemoryInfo(self.handle)
            gpu_usage = info.used / info.total * 100
            return f"{gpu_usage:.1f}%"
        except:
            return "N/A"
        
    def _get_system_info(self):
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        return f"CPU: {cpu_percent}% | RAM: {memory.percent}% | GPU: {self._get_gpu_info()}"

    def on_train_begin(self, args, state, control, **kwargs):
        self.training_start = time.time()
        console.print(Panel.fit(
            "[bold green]开始训练[/bold green]",
            title="Training Start"
        ))
        self.task = self.progress.add_task("[cyan]Training...", total=args.num_train_epochs)
        self.progress.start()

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start = time.time()
        console.print(f"\n[bold blue]Epoch {state.epoch + 1}/{args.num_train_epochs}[/bold blue]")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            # 创建一个表格来显示训练指标
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Metric")
            table.add_column("Value")
            
            # 添加基本指标
            for key, value in logs.items():
                if isinstance(value, float):
                    table.add_row(key, f"{value:.4f}")
                else:
                    table.add_row(key, str(value))
            
            # 添加系统信息
            table.add_row("System Info", self._get_system_info())
            
            # 如果有进度信息，更新进度条
            if "epoch" in logs:
                self.progress.update(self.task, completed=logs["epoch"])
            
            console.print(table)

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch_time = time.time() - self.epoch_start
        console.print(f"[green]Epoch完成! 耗时: {epoch_time:.2f}秒[/green]")

    def on_train_end(self, args, state, control, **kwargs):
        self.progress.stop()
        total_time = time.time() - self.training_start
        console.print(Panel.fit(
            f"[bold green]训练完成![/bold green]\n"
            f"总耗时: {total_time:.2f}秒\n"
            f"最终loss: {state.log_history[-1].get('loss', 'N/A')}\n"
            f"保存路径: {args.output_dir}",
            title="Training Complete"
        ))
        nvml.nvmlShutdown() 