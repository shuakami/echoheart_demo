from dataclasses import dataclass

@dataclass
class TrainingConfig:
    # 模型配置
    model_name: str = "Qwen/Qwen1.5-0.5B"  # 模型名称
    train_dataset_path: str = "data/converted_dataset.json"  # 训练数据集路径
    output_dir: str = "output/qwen-ft"  # 输出目录

    # 训练参数
    num_train_epochs: int = 1  # 训练轮数
    per_device_train_batch_size: int = 2  # 每个设备的批次大小
    gradient_accumulation_steps: int = 4  # 梯度累积步数
    learning_rate: float = 2e-5  # 学习率
    weight_decay: float = 0.01  # 权重衰减
    max_grad_norm: float = 1.0  # 梯度裁剪
    
    # 保存和评估
    save_steps: int = 50  # 每多少步保存一次
    eval_steps: int = 100  # 每多少步评估一次
    logging_steps: int = 10  # 每多少步记录一次日志
    
    # 其他配置
    seed: int = 42  # 随机种子
    fp16: bool = True  # 是否使用混合精度训练
    
config = TrainingConfig() 