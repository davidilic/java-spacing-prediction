from dataclasses import dataclass

@dataclass
class TrainingConfig:
    """Configuration for model training."""
    num_epochs: int = 10
    batch_size: int = 32
    initial_lr: float = 5e-5
    max_lr: float = 2e-4
    weight_decay: float = 0.04
    space_loss_weight: float = 1.0
    newline_loss_weight: float = 1.0
    train_split: float = 0.8