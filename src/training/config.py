from dataclasses import dataclass

@dataclass
class TrainingConfig:

    model_name: str

    learning_rate: float = 2e-5

    batch_size: int = 16

    epochs: int = 3

    max_length: int = 512