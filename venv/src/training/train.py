import logging
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim import AdamW
from sklearn.preprocessing import LabelEncoder


# Logging setup
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("training.log"), logging.StreamHandler()],
    )


# Prepare the dataset from CSV
class FAQDataset(Dataset):
    def __init__(self, queries, responses, tokenizer, max_length):
        self.queries = queries
        self.responses = responses
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        query = self.queries[idx]
        response = self.responses[idx]

        # Tokenizing the queries and responses for causal LM (text generation)
        inputs = self.tokenizer(
            query,
            response,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": inputs["input_ids"].flatten(),
            "attention_mask": inputs["attention_mask"].flatten(),
        }


# Main training pipeline
def train_model(data_path, model_name="facebook/opt-2.7b", max_length=512, batch_size=8, epochs=3, learning_rate=5e-5):
    # Step 1: Load and preprocess data
    logging.info("Loading data...")
    df = pd.read_csv(data_path)

    # Ensure CSV has the required columns
    if "query" not in df.columns or "response" not in df.columns:
        raise ValueError("CSV must have 'query' and 'response' columns")

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Create dataset
    dataset = FAQDataset(df["query"].tolist(), df["response"].tolist(), tokenizer, max_length)

    # Split dataset into train and validation sets (80% train, 20% validation)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Step 2: Initialize model
    logging.info("Initializing model...")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)  # Move the model to the device (GPU or CPU)

    # Step 3: Define optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Step 4: Train the model
    logging.info("Training the model...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            total_loss += loss.item()

            # Backpropagation
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(train_loader)
        logging.info(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")

        # Step 5: Evaluate on validation set
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                # Forward pass
                outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
                loss = outputs.loss
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        logging.info(f"Validation Loss: {avg_val_loss:.4f}")

    logging.info("Training completed.")


# Main function
if __name__ == "__main__":
    # Set device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    setup_logging()
    data_path = r"C:\Users\amrey\Desktop\faq_data.csv"  # Your CSV path
    train_model(data_path)
