import time
import psutil
import torch
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn
from rich.table import Table
from rich.panel import Panel
from transformers import TrainerCallback

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
        self.gpu_monitoring_available = False # 禁用GPU监控
        console.print("[bold yellow]提示：GPU 监控已禁用。[/bold yellow]")
        
    def _get_system_info(self):
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        return f"CPU: {cpu_percent}% | RAM: {memory.percent}%"

    def on_train_begin(self, args, state, control, **kwargs):
        self.training_start = time.time()
        console.print(Panel.fit(
            "[bold green]开始训练[/bold green]",
            title="Training Start"
        ))
        # Trainer state max_steps might not be ready at train_begin, calculate it
        num_update_steps_per_epoch = len(control.model.dataset) // args.gradient_accumulation_steps
        max_steps = int(args.num_train_epochs * num_update_steps_per_epoch)
        self.task = self.progress.add_task("[cyan]Training...", total=max_steps)
        self.progress.start()

    def on_step_end(self, args, state, control, **kwargs):
        # 更新全局进度条
        self.progress.update(self.task, completed=state.global_step)
        pass 

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and state.is_local_process_zero: # 仅在主进程记录
            table = Table(show_header=True, header_style="bold magenta", title=f"Step {state.global_step}")
            table.add_column("Metric", style="dim")
            table.add_column("Value")
            
            for key, value in logs.items():
                if key not in ["epoch", "step"]:
                    if isinstance(value, float):
                        table.add_row(key, f"{value:.4f}")
                    else:
                        table.add_row(key, str(value))
            
            table.add_row("Epoch", f"{state.epoch:.2f}")
            table.add_row("System Info", self._get_system_info())
            
            console.print(table)

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start = time.time()
        if state.is_local_process_zero:
             console.print(f"\n[bold blue]---> 开始 Epoch {int(state.epoch) + 1}/{int(args.num_train_epochs)} <---[/bold blue]")

    def on_epoch_end(self, args, state, control, **kwargs):
        if state.is_local_process_zero:
            epoch_time = time.time() - self.epoch_start
            console.print(f"[green]---> Epoch {int(state.epoch)+1} 完成! 耗时: {epoch_time:.2f}秒 <---[/green]")

    def on_train_end(self, args, state, control, **kwargs):
        self.progress.stop()
        if state.is_local_process_zero:
            total_time = time.time() - self.training_start
            final_loss = state.log_history[-1].get('loss', 'N/A') if state.log_history else 'N/A'
            
            console.print(Panel.fit(
                f"[bold green]训练完成![/bold green]\n"
                f"总耗时: {total_time:.2f}秒\n"
                f"总步数: {state.global_step}\n"
                f"最终loss: {final_loss}\n"
                f"保存路径: {args.output_dir}",
                title="Training Complete"
            )) 