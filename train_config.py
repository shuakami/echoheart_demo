from dataclasses import dataclass

@dataclass
class TrainingConfig:
    # 模型配置
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"  # 选择的模型
    train_dataset_path: str = "data/converted_dataset.json"  # 训练数据集路径
    output_dir: str = "output/qwen2-1.5b-qlora-ft"  # 输出目录 (反映QLoRA)

    # 训练参数 (为 QLoRA 调整)
    num_train_epochs: int = 2  # 训练轮数 
    per_device_train_batch_size: int = 2  # 每个设备的批次大小 (尝试调回 2)
    gradient_accumulation_steps: int = 4  # 梯度累积步数 (调回 4)
    learning_rate: float = 2e-4  # LoRA 训练通常使用稍高的学习率 (例如 1e-4 或 2e-4)
    weight_decay: float = 0.01  # 权重衰减
    max_grad_norm: float = 1.0  # 梯度裁剪
    
    # 保存和评估
    # 步数会变化: steps_per_epoch = ceil(207 / (2*4)) = 26. max_steps = 26 * 2 = 52
    save_steps: int = 25  # 每多少步保存一次 (调整以适应新步数)
    eval_steps: int = 100 # 评估步数
    logging_steps: int = 5 # 每多少步记录一次日志 (更频繁些)
    
    # 其他配置
    seed: int = 42  # 随机种子
    fp16: bool = True  # 是否使用混合精度训练
    # bf16 会在 train.py 中根据 GPU 支持自动启用
    
config = TrainingConfig() 