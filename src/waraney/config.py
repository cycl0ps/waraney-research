from dataclasses import dataclass


@dataclass
class ModelConfig:
    model_name: str = "bert-base-uncased"
    max_length: int = 256


@dataclass
class TrainingConfig:
    batch_size: int = 16
    epochs: int = 5
    learning_rate: float = 2e-5