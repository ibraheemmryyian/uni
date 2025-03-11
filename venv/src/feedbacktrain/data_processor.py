import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple
import numpy as np
from transformers import AutoTokenizer

class CustomerSupportDataset(Dataset):
    def __init__(self, queries: List[str], responses: List[str], feedback: List[float], tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer
        self.feedback = feedback  # Store feedback ratings
        self.encodings = self._encode_pairs(queries, responses)
        
    def _encode_pairs(self, queries: List[str], responses: List[str]) -> Dict:
        """Encode query-response pairs."""        
        formatted_inputs = [
            f"Query: {query} Response: {response}"
            for query, response in zip(queries, responses)
        ]
        
        return self.tokenizer(
            formatted_inputs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
    def __len__(self) -> int:
        return len(self.encodings['input_ids'])
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': torch.tensor(self.feedback[idx], dtype=torch.float)  # Feed feedback as target label
        }

def prepare_dataloaders(
    queries: List[str],
    responses: List[str],
    feedback: List[float],  # Feedback ratings are passed here
    tokenizer: AutoTokenizer,
    config: Dict
) -> Tuple[DataLoader, DataLoader]:
    """Prepare training and validation dataloaders."""
    
    # Split data
    indices = np.random.permutation(len(queries))
    split = int(len(queries) * (1 - config['training']['validation_split']))
    
    train_indices = indices[:split]
    val_indices = indices[split:]
    
    # Create datasets
    train_dataset = CustomerSupportDataset(
        [queries[i] for i in train_indices],
        [responses[i] for i in train_indices],
        [feedback[i] for i in train_indices],
        tokenizer
    )
    
    val_dataset = CustomerSupportDataset(
        [queries[i] for i in val_indices],
        [responses[i] for i in val_indices],
        [feedback[i] for i in val_indices],
        tokenizer
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size']
    )
    
    return train_loader, val_loader
