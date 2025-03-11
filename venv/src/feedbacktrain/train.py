import pandas as pd
from transformers import AutoTokenizer
from data_processor import prepare_dataloaders  # Make sure this is the correct import path for your data processor
from config import TRAINING_CONFIG
from trainer import ModelTrainer
from transformers import AutoModelForSequenceClassification
import torch

# Load the CSV file (feedback.csv)
df = pd.read_csv("feedback.csv")  # Replace with the correct path if necessary

# Extract queries, responses, and feedback ratings
queries = df['query'].tolist()
responses = df['response'].tolist()
feedback = df['feedback'].tolist()

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-2.7b")

# Prepare the dataloaders (train and validation sets)
train_loader, val_loader = prepare_dataloaders(queries, responses, feedback, tokenizer, TRAINING_CONFIG)

# Initialize the model (this is a regression task, so we need a model for sequence classification with one output)
model = AutoModelForSequenceClassification.from_pretrained("facebook/opt-2.7b", num_labels=1)  # num_labels=1 for regression task

# Initialize the trainer
trainer = ModelTrainer(model, TRAINING_CONFIG)

# Start training the model
trainer.train(train_loader, val_loader)
