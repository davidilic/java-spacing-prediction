from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Configuration for the transformer model."""
    vocab_size: int
    type_vocab_size: int
    max_newlines: int
    d_model: int = 256
    nhead: int = 8
    num_encoder_layers: int = 6
    dim_feedforward: int = 1024
    dropout: float = 0.1
    max_seq_length: int = 2048
