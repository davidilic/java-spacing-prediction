from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional
import json
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from dataset.types import TokenSample, BatchedSamples

class TokenDataset(Dataset):
    def __init__(self, file_path: Path, existing_vocab: Optional[Dict] = None, existing_types: Optional[Dict] = None):
        if existing_vocab and existing_types:
            self.vocab = VocabularyBuilder()
            self.vocab.token_to_idx = existing_vocab.copy() 
            self.vocab.type_to_idx = existing_types.copy()
            
            self.vocab.token_to_idx['<UNK>'] = len(existing_vocab) - 1
            self.vocab.type_to_idx['<UNK>'] = existing_types['IDENTIFIER']
        else:
            self.vocab = VocabularyBuilder()
            
        self.samples: List[Dict] = []
        self.unk_token_count = 0
        self.total_token_count = 0
        
        self._load_data(file_path)
        
        if not existing_vocab:
            self.vocab.finalize()
            
        if self.total_token_count > 0:
            print(f"Unknown token ratio: {self.unk_token_count / self.total_token_count:.2%}")

    def _process_sample(self, data: Dict) -> Dict:
        tokens = []
        token_types = []
        
        for t in data['tokens']:
            self.total_token_count += 1
            if t not in self.vocab.token_to_idx:
                self.unk_token_count += 1
                tokens.append(self.vocab.token_to_idx['<UNK>'])
            else:
                tokens.append(self.vocab.token_to_idx[t])
                
        for tt in data['token_types']:
            if tt not in self.vocab.type_to_idx:
                token_types.append(self.vocab.type_to_idx['<UNK>'])
            else:
                token_types.append(self.vocab.type_to_idx[tt])
                
        return {
            'tokens': tokens,
            'token_types': token_types,
            'spaces': data['spaces'],
            'newlines': [data['newlines']] * len(tokens),
            'scope_depth': [data['scope_depth']] * len(tokens),
            'file_id': data['file_id']
        }
    
    def _load_data(self, file_path: Path) -> None:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                sample = self._process_sample(json.loads(line))
                self.samples.append(sample)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> TokenSample:
        sample = self.samples[idx]
        return TokenSample(
            tokens=torch.tensor(sample['tokens'], dtype=torch.long),
            token_types=torch.tensor(sample['token_types'], dtype=torch.long),
            spaces=torch.tensor(sample['spaces'], dtype=torch.float),
            newlines=torch.tensor(sample['newlines'], dtype=torch.long),
            scope_depth=torch.tensor(sample['scope_depth'], dtype=torch.float),
            file_id=sample['file_id']
        )

class VocabularyBuilder:
    """Builds and manages vocabulary for tokens and types."""
    
    def __init__(self):
        self.token_to_idx = defaultdict(lambda: len(self.token_to_idx))
        self.type_to_idx = defaultdict(lambda: len(self.type_to_idx))
        self._add_special_tokens()
    
    def _add_special_tokens(self) -> None:
        self.pad_token_idx = self.token_to_idx['<PAD>']
        self.pad_type_idx = self.type_to_idx['<PAD>']
    
    def finalize(self) -> None:
        self.token_to_idx = dict(self.token_to_idx)
        self.type_to_idx = dict(self.type_to_idx)
    
class DataCollator:
    """Collates samples into batches with padding."""
    
    def __call__(self, samples: List[TokenSample]) -> BatchedSamples:
        tokens = [s.tokens for s in samples]
        token_types = [s.token_types for s in samples]
        spaces = [s.spaces for s in samples]
        newlines = [s.newlines for s in samples]
        scope_depths = [s.scope_depth for s in samples]
        
        tokens_padded = pad_sequence(tokens, batch_first=True, padding_value=0)
        attention_mask = (tokens_padded != 0).float()
        
        return BatchedSamples(
            tokens=tokens_padded,
            token_types=pad_sequence(token_types, batch_first=True, padding_value=0),
            spaces=pad_sequence(spaces, batch_first=True, padding_value=0),
            newlines=pad_sequence(newlines, batch_first=True, padding_value=0),
            scope_depth=pad_sequence(scope_depths, batch_first=True, padding_value=0),
            attention_mask=attention_mask
        )