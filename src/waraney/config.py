from dataclasses import dataclass

@dataclass
class ModelConfig:
    model_name: str
    max_length: int = 256