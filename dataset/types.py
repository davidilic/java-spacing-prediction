from enum import Enum
from dataclasses import dataclass
from typing import List
import torch

@dataclass
class TokenSample:
    """Single token sample with its associated metadata."""
    tokens: torch.Tensor
    token_types: torch.Tensor
    spaces: torch.Tensor
    newlines: torch.Tensor
    scope_depth: torch.Tensor
    file_id: str

@dataclass
class BatchedSamples:
    """Batch of samples with padding and attention mask."""
    tokens: torch.Tensor
    token_types: torch.Tensor
    spaces: torch.Tensor
    newlines: torch.Tensor
    scope_depth: torch.Tensor
    attention_mask: torch.Tensor

class TokenType(Enum):
    STRING = 'STRING'
    IDENTIFIER = 'IDENTIFIER' 
    DELIMITER = 'DELIMITER'
    OPERATOR = 'OPERATOR'
    WHITESPACE = 'WHITESPACE'
    KEYWORD = 'KEYWORD'

@dataclass
class CodeInstance:
    file_id: str
    tokens: List[str]
    token_types: List[str]
    spaces: List[int]
    newlines: int
    scope_depth: int