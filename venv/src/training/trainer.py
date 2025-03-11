import torch
from torch import nn
from transformers import get_linear_schedule_with_warmup
from typing import Dict, Any
import logging
from tqdm import tqdm

class ModelTrainer:
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        self.model = model
        self.config = config
        
        # Force the device to CPU
        self.device = torch.device('cpu')
        self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate']
        )
        
    def train(self, train_loader, val_loader):
        """Train the model"""
        num_training_steps = len(train_loader) * self.config['training']['epochs']
        
        # Initialize scheduler
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config['training']['warmup_steps'],
            num_training_steps=num_training_steps
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config['training']['epochs']):
            # Training phase
            self.model.train()
            train_loss = 0
            
            with tqdm(train_loader, desc=f'Epoch {epoch+1}') as pbar:
                for batch in pbar:
                    # Move batch to CPU (force CPU)
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    
                    # Forward pass
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    
                    # Backward pass
                    loss.backward()
                    
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['max_grad_norm']
                    )
                    
                    # Update weights
                    self.optimizer.step()
                    scheduler.step()
                    self.optimizer.zero_grad()
                    
                    train_loss += loss.item()
                    pbar.set_postfix({'loss': loss.item()})
            
            # Validation phase
            val_loss = self._validate(val_loader)
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self._save_checkpoint('best_model.pt')
            else:
                patience_counter += 1
                
            if patience_counter >= self.config['training']['early_stopping_patience']:
                logging.info('Early stopping triggered')
                break
                
            logging.info(f'Epoch {epoch+1}: train_loss={train_loss/len(train_loader):.4f}, val_loss={val_loss:.4f}')
    
    def _validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Ensure batch is moved to CPU
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                val_loss += outputs.loss.item()
                
        return val_loss / len(val_loader)
    
    def _save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, filename)
