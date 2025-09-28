from config import *
from train_utils import TrainerUtils
from tqdm import tqdm
import torch
import os

class Trainer:
    def __init__(self):
        self.trainer_utils = TrainerUtils()
        self.model, self.train_dataloader, self.val_dataloader, self.optimizer, self.scheduler = self.trainer_utils.load_train_utils()
        self.model = self.model.to(self.trainer_utils.device)
        self.epoch_num = 0
        self.step_num = 0
        self.num_epochs = NUM_EPOCHS
        self.model_save_path = MODEL_SAVE_PATH
        self.best_val_accuracy = 0.0

    def train(self):
        print(f"Starting training for {self.num_epochs} epochs...")
        print(f"Training samples: {len(self.train_dataloader.dataset)}")
        print(f"Validation samples: {len(self.val_dataloader.dataset)}")

        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.num_epochs}")

            # Training phase
            self.model.train()
            epoch_train_loss = 0
            num_batches = 0

            for data in tqdm(self.train_dataloader, desc=f"Training epoch {epoch + 1}"):
                loss, self.model, self.optimizer = self.trainer_utils.train_one_step(
                    data, self.model, self.optimizer, self.scheduler
                )

                epoch_train_loss += loss
                num_batches += 1
                self.step_num += 1

            # Calculate average training loss
            avg_train_loss = epoch_train_loss / num_batches

            # Validation phase
            print("\n\nRunning validation...")
            val_loss, val_accuracy, val_predictions, val_labels = self.trainer_utils.validate(
                self.model, self.val_dataloader
            )

            # Print epoch results
            print(f"Training Loss: {avg_train_loss:.4f}")
            print(f"Validation Loss: {val_loss:.4f}")
            print(f"Validation Accuracy: {val_accuracy:.4f}")

            # Save best model
            if val_accuracy > self.best_val_accuracy:
                self.best_val_accuracy = val_accuracy
                # Construct the full path for saving the best model
                best_model_path = os.path.join(self.model_save_path, f"{self.model_save_path}_best_epoch_{epoch + 1}.pt")
                saved_message = self.trainer_utils.save_checkpoint(
                    self.model, self.optimizer, epoch + 1, val_loss, best_model_path
                )
                print(f"\nNew best model! {saved_message}")

            # Save checkpoint every 2 epochs
            if (epoch + 1) % 2 == 0:
                # Construct the full path for saving the regular checkpoint
                checkpoint_path = os.path.join(self.model_save_path, f"{self.model_save_path}_epoch_{epoch + 1}.pt")
                saved_message = self.trainer_utils.save_checkpoint(
                    self.model, self.optimizer, epoch + 1, val_loss, checkpoint_path
                )
                print(saved_message)


            self.epoch_num += 1

        print(f"\nTraining completed! Best validation accuracy: {self.best_val_accuracy:.4f}")

if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
