from typing import Dict, Any
from pathlib import Path
import torch
from datetime import datetime
from model.transformer import TransformerModel
from dataset.dataset import TokenDataset

class CheckpointManager:
    """Manages model checkpointing with complete state saving."""
    
    def __init__(self, save_dir: Path):
        self.save_dir = save_dir
        self.best_val_loss = float('inf')
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def save_if_best(self, model: TransformerModel, optimizer: torch.optim.Optimizer, 
                     dataset: TokenDataset, metrics: Dict[str, float], epoch: int) -> bool:
        
        current_loss = metrics['loss']
        if current_loss >= self.best_val_loss:
            return False
            
        self.best_val_loss = current_loss
        self._save_checkpoint(model, optimizer, dataset, metrics, epoch)
        return True
    
    def _save_checkpoint(self, model: TransformerModel, optimizer: torch.optim.Optimizer, 
                         dataset: TokenDataset, metrics: Dict[str, float], epoch: int) -> None:
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        loss_str = f"{metrics['loss']:.4f}".replace('.', '_')
        filename = f'checkpoint_loss_{loss_str}_{timestamp}.pt'
        path = self.save_dir / filename
        
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'best_val_loss': self.best_val_loss,
            'model_config': model.config.__dict__,
            'token_to_idx': dataset.vocab.token_to_idx,
            'type_to_idx': dataset.vocab.type_to_idx,
            'pad_token_idx': dataset.vocab.pad_token_idx,
            'pad_type_idx': dataset.vocab.pad_type_idx,
            'metrics': metrics,
        }
        
        torch.save(checkpoint, path)
        print(f"New best model (loss: {metrics['loss']:.4f}) saved to {path}")
    
    @staticmethod
    def load_checkpoint(checkpoint_path: Path, device: torch.device) -> Dict[str, Any]:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        return checkpoint