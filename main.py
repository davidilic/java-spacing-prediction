from pathlib import Path
import torch
from dataset.dataset import TokenDataset
from model.transformer import TransformerModel
from model.config import ModelConfig
from training.trainer import ModelTrainer, create_data_loaders, get_dataset_stats
from training.config import TrainingConfig
from training.checkpointing import CheckpointManager


def main():
    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset = TokenDataset(Path('./dataset/data/train.jsonl'))
    _, max_newlines = get_dataset_stats(dataset)
    
    training_config = TrainingConfig(
        batch_size=16,
        initial_lr=1e-4,
        weight_decay=0.05,
        space_loss_weight=0.7,
        newline_loss_weight=0.3,
        train_split=0.8,
    )
    
    model_config = ModelConfig(
        vocab_size=len(dataset.vocab.token_to_idx),
        type_vocab_size=len(dataset.vocab.type_to_idx),
        max_newlines=max_newlines,
        d_model=384,
        nhead=6,    
        num_encoder_layers=6,
        dim_feedforward=1024,
        dropout=0.15
    )
    
    model = TransformerModel(model_config).to(device)
    trainer = ModelTrainer(model, training_config, device)
    checkpoint_manager = CheckpointManager(Path('./checkpoints'))
    
    train_loader, val_loader = create_data_loaders(dataset, training_config)
    trainer.setup_scheduler(len(train_loader))
    
    for epoch in range(training_config.num_epochs):
        train_metrics = trainer.train_epoch(train_loader)
        val_metrics = trainer.validate(val_loader)
        
        print(f'Epoch {epoch+1}/{training_config.num_epochs}:')
        print(f'Training Loss: {train_metrics["loss"]:.4f}')
        print(f'Training Space MSE: {train_metrics["space_mse"]:.4f}')
        print(f'Training Newline Accuracy: {train_metrics["newline_accuracy"]:.2%}')
        print(f'Validation Loss: {val_metrics["loss"]:.4f}')
        print(f'Validation Space MSE: {val_metrics["space_mse"]:.4f}')
        print(f'Validation Newline Accuracy: {val_metrics["newline_accuracy"]:.2%}')
        
        checkpoint_manager.save_if_best(model, trainer.optimizer, dataset, val_metrics, epoch)

if __name__ == "__main__":
    main()