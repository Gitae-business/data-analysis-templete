import os
import torch
import numpy as np
from config import config

class EarlyStopping:
    def __init__(self, patience=5, verbose=True, delta=0.0, model_name='MLP', fold=0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_loss = np.inf
        self.counter = 0
        self.early_stop = False
        self.model_name = model_name
        self.fold = fold

        self.checkpoint_dir = os.path.join(config.CHECKPOINT_DIR, model_name)
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def __call__(self, val_loss, model, optimizer=None, epoch=None):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0

            best_path = os.path.join(self.checkpoint_dir, f"{self.model_name}_fold_{self.fold}.pth")
            torch.save({
                'epoch': epoch,
                'val_loss': val_loss,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
                'early_stop_counter': self.counter
            }, best_path)

            if self.verbose:
                print(f"Validation loss improved. Best model saved to {best_path}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"No improvement in validation loss. Counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("Early stopping triggered.")
