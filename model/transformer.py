from typing import Tuple
import torch
from model.config import ModelConfig
from torch import nn

class EmbeddingLayer(nn.Module):
    """Combines different embeddings for input tokens."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=0)
        self.type_embedding = nn.Embedding(config.type_vocab_size, config.d_model, padding_idx=0)
        self.scope_embedding = nn.Linear(1, config.d_model)
        self.pos_encoder = nn.Embedding(config.max_seq_length, config.d_model)
        
    def forward(self, tokens: torch.Tensor, token_types: torch.Tensor, 
                scope_depth: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = tokens.size()
        positions = torch.arange(seq_len, device=tokens.device).unsqueeze(0).expand(batch_size, -1)
        
        embeddings = (
            self.token_embedding(tokens) +
            self.type_embedding(token_types) +
            self.scope_embedding(scope_depth.unsqueeze(-1)) +
            self.pos_encoder(positions)
        )
        return embeddings

class OutputLayer(nn.Module):
    """Handles space regression and newline classification outputs."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.spaces_output = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, 1)
        )
        self.newlines_output = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, config.max_newlines + 1)
        )
    
    def forward(self, transformer_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        space_output = self.spaces_output(transformer_output).squeeze(-1)
        newline_logits = self.newlines_output(transformer_output)
        return space_output, newline_logits

class TransformerModel(nn.Module):
    """Transformer model for code formatting prediction."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.embedding_layer = EmbeddingLayer(config)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            config.num_encoder_layers
        )
        
        self.output_layer = OutputLayer(config)
    
    def forward(self, tokens: torch.Tensor, token_types: torch.Tensor, 
                scope_depth: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        embeddings = self.embedding_layer(tokens, token_types, scope_depth)
        
        key_padding_mask = (attention_mask == 0)
        transformer_output = self.transformer_encoder(
            embeddings, 
            src_key_padding_mask=key_padding_mask
        )
        
        return self.output_layer(transformer_output)

