import yaml
from dataclasses import dataclass


@dataclass
class ModelConfig:
    backbone: str
    num_labels: int


@dataclass
class TrainingConfig:
    batch_size: int
    epochs: int
    learning_rate: float


def load_config(path: str):

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    return config