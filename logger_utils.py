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

    def on_train_begin(self, args, state, control, model=None, train_dataloader=None, **kwargs):
        self.training_start = time.time()
        console.print(Panel.fit(
            "[bold green]开始训练[/bold green]",
            title="Training Start"
        ))
        
        max_steps = 1 # Default value
        if train_dataloader is not None:
            try:
                num_examples = len(train_dataloader.dataset)
                # Correct calculation for max_steps
                steps_per_epoch = (num_examples + args.train_batch_size * args.gradient_accumulation_steps - 1) // (args.train_batch_size * args.gradient_accumulation_steps)
                max_steps = int(steps_per_epoch * args.num_train_epochs)
                console.print(f"[dim]计算得到的总步数: {max_steps} (数据集: {num_examples}, batch: {args.train_batch_size}, grad_acc: {args.gradient_accumulation_steps}, epochs: {args.num_train_epochs})[/dim]")
            except Exception as e:
                 console.print(f"[dim yellow]警告: 计算总步数时出错 ({e})，将使用 state.max_steps。[/dim yellow]")
                 max_steps = state.max_steps 
        else:
             console.print(f"[dim yellow]警告: 无法访问 train_dataloader，将使用 state.max_steps ({state.max_steps})。[/dim yellow]")
             max_steps = state.max_steps 
             
        if max_steps <= 0: # Fallback if calculation fails or epochs=0
            max_steps = 1 
            console.print(f"[dim yellow]警告: 总步数计算结果无效，设置为 1。[/dim yellow]")

        self.task = self.progress.add_task("[cyan]Training...", total=max_steps)
        self.progress.start()

    def on_step_end(self, args, state, control, **kwargs):
        # 更新全局进度条, 确保不会超过total
        if self.task is not None and state.global_step <= self.progress.tasks[self.task].total:
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
            # Ensure epoch number is displayed correctly (it's 0-indexed internally at the end)
            epoch_display = int(state.epoch)
            console.print(f"[green]---> Epoch {epoch_display} 完成! 耗时: {epoch_time:.2f}秒 <---[/green]")

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