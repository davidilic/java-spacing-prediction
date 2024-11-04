from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from dataset.dataset import TokenDataset, BatchedSamples, DataCollator
from model.transformer import TransformerModel
from training.config import TrainingConfig

class MetricsTracker:
    """Tracks and computes training metrics."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.loss = 0
        self.space_mse = 0
        self.newline_correct = 0
        self.total_tokens = 0
        self.num_batches = 0
    
    def update(self, loss: float, space_output: torch.Tensor, spaces: torch.Tensor,
              newline_preds: torch.Tensor, newlines: torch.Tensor, 
              attention_mask: torch.Tensor):
        self.loss += loss
        self.space_mse += ((space_output - spaces) ** 2 * attention_mask).sum().item()
        self.newline_correct += ((newline_preds == newlines).float() * attention_mask).sum().item()
        self.total_tokens += attention_mask.sum().item()
        self.num_batches += 1
    
    def compute(self) -> Dict[str, float]:
        return {
            'loss': self.loss / self.num_batches,
            'space_mse': self.space_mse / self.total_tokens,
            'newline_accuracy': self.newline_correct / self.total_tokens
        }

class ModelTrainer:
    """Handles model training and validation."""
    
    def __init__(self, model: TransformerModel, config: TrainingConfig, device: torch.device):
        self.model = model
        self.config = config
        self.device = device
        self.setup_training()
        
    def setup_training(self):
        self.mse_criterion = nn.MSELoss(reduction='none')
        self.ce_criterion = nn.CrossEntropyLoss(reduction='none')
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.initial_lr,
            weight_decay=self.config.weight_decay
        )
        
    def setup_scheduler(self, steps_per_epoch: int):
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config.max_lr,
            epochs=self.config.num_epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.1,
            div_factor=25,
            final_div_factor=1e4,
        )
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        metrics = MetricsTracker()
        
        for batch in train_loader:
            metrics.update(*self.process_batch(batch, is_training=True))
            
        return metrics.compute()
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        metrics = MetricsTracker()
        
        with torch.no_grad():
            for batch in val_loader:
                metrics.update(*self.process_batch(batch, is_training=False))
                
        return metrics.compute()
    
    def process_batch(self, batch: BatchedSamples, is_training: bool) -> Tuple:
        """Process a single batch of data."""
        batch_on_device = self._prepare_batch(batch)
        
        model_inputs = {
            'tokens': batch_on_device['tokens'],
            'token_types': batch_on_device['token_types'],
            'scope_depth': batch_on_device['scope_depth'],
            'attention_mask': batch_on_device['attention_mask']
        }
        
        space_output, newline_logits = self.model(**model_inputs)
        
        loss, space_output, newline_preds = self._compute_loss_and_predictions(
            space_output, newline_logits, batch_on_device
        )
        
        if is_training:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            
        return (
            loss.item(),
            space_output,
            batch_on_device['spaces'],
            newline_preds,
            batch_on_device['newlines'],
            batch_on_device['attention_mask']
        )
    
    def _prepare_batch(self, batch: BatchedSamples) -> Dict[str, torch.Tensor]:
        return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.__dict__.items()}
    
    def _compute_loss_and_predictions(self, space_output: torch.Tensor,
                                    newline_logits: torch.Tensor,
                                    batch: Dict[str, torch.Tensor]) -> Tuple:
        space_loss = self.mse_criterion(space_output, batch['spaces'])
        newline_loss = self.ce_criterion(
            newline_logits.view(-1, self.model.config.max_newlines + 1),
            batch['newlines'].long().view(-1)
        ).view_as(space_loss)
        
        losses = self._apply_loss_masks(space_loss, newline_loss, batch['attention_mask'])
        total_loss = (
            self.config.space_loss_weight * losses[0] +
            self.config.newline_loss_weight * losses[1]
        )
        
        return total_loss, space_output, newline_logits.argmax(dim=-1)
    
    def _apply_loss_masks(self, space_loss: torch.Tensor,
                         newline_loss: torch.Tensor,
                         attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        space_loss = (space_loss * attention_mask).sum() / attention_mask.sum()
        newline_loss = (newline_loss * attention_mask).sum() / attention_mask.sum()
        return space_loss, newline_loss


def create_data_loaders(dataset: TokenDataset, config: TrainingConfig) -> Tuple[DataLoader, DataLoader]:
    """Creates train and validation data loaders."""
    file_ids = list(set(sample['file_id'] for sample in dataset.samples))
    train_size = int(len(file_ids) * config.train_split)
    
    train_files = set(file_ids[:train_size])
    train_indices = [i for i, sample in enumerate(dataset.samples) 
                    if sample['file_id'] in train_files]
    val_indices = [i for i, sample in enumerate(dataset.samples) 
                  if sample['file_id'] not in train_files]
    
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    
    collator = DataCollator()
    
    return (
        DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            collate_fn=collator
        ),
        DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=collator
        )
    )

def get_dataset_stats(dataset: TokenDataset) -> Tuple[int, int]:
    """Calculates maximum spaces and newlines from dataset."""
    max_spaces = max_newlines = 0
    
    for sample in dataset:
        if sample.spaces.numel() > 0:
            max_spaces = max(max_spaces, sample.spaces.max().item())
        if sample.newlines.numel() > 0:
            max_newlines = max(max_newlines, sample.newlines.max().item())
    
    if max_spaces == 0 or max_newlines == 0:
        raise ValueError("Could not find valid max values in dataset")
    
    return int(max_spaces), int(max_newlines)
