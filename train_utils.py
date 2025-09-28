import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from config import *
from transformer_classifier import build_transformer_model
from dataset import EmotionClassificationDataset
from tokenizer import tokenizer
import os
from sklearn.metrics import accuracy_score
import numpy as np

class TrainerUtils:

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

    def load_train_utils(self):
        """Load model, datasets, dataloaders, optimizer and writer"""

        # Build model
        model = build_transformer_model()

        # Create datasets
        train_dataset = EmotionClassificationDataset("train", tokenizer, MAX_SEQ_LEN)
        val_dataset = EmotionClassificationDataset("validation", tokenizer, MAX_SEQ_LEN)

        # Create dataloaders
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            drop_last= True,
            num_workers=2,
            pin_memory=True
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            drop_last= True,
            num_workers=2,
            pin_memory=True
        )


        # Create optimizer
        optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)

        # Create learning rate scheduler
        total_steps = len(train_dataloader) * NUM_EPOCHS
        scheduler = OneCycleLR(
            optimizer,
            max_lr=LR,
            total_steps=total_steps,
            pct_start=0.1
        )

        return model, train_dataloader, val_dataloader, optimizer, scheduler

    def train_one_step(self, data, model, optimizer, scheduler=None):
        """Train one step"""
        model.train()

        # Move data to device
        input_ids = data["input_ids"].to(self.device)
        attention_mask = data["attention_mask"].to(self.device)
        labels = data["labels"].to(self.device)

        # Forward pass
        outputs = model(input_ids, attention_mask)

        # Calculate loss
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        if scheduler:
            scheduler.step()

        return loss.item(), model, optimizer

    def validate(self, model, val_dataloader):
        """Validate model"""
        model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []

        loss_fn = nn.CrossEntropyLoss()

        with torch.no_grad():
            for data in val_dataloader:
                input_ids = data["input_ids"].to(self.device)
                attention_mask = data["attention_mask"].to(self.device)
                labels = data["labels"].to(self.device)

                outputs = model(input_ids, attention_mask)
                loss = loss_fn(outputs, labels)

                total_loss += loss.item()

                # Get predictions
                predictions = torch.argmax(outputs, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(val_dataloader)
        accuracy = accuracy_score(all_labels, all_predictions)

        return avg_loss, accuracy, all_predictions, all_labels

    def save_checkpoint(self, model, optimizer, epoch, loss, save_path):
        """Save model checkpoint"""
        # Ensure the directory exists before saving
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }

        torch.save(checkpoint, save_path)
        return f"Model saved at {save_path}"

    def load_checkpoint(self, model, optimizer, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        return model, optimizer, epoch, loss
