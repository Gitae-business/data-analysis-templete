import os
import torch
import numpy as np

class EarlyStopping:
    def __init__(self, patience=5, verbose=True, delta=0.0, checkpoint_dir=None, model_name='model'):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_loss = np.inf
        self.counter = 0
        self.early_stop = False
        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name
        if self.checkpoint_dir:
            os.makedirs(self.checkpoint_dir, exist_ok=True)

    def __call__(self, val_loss, model, epoch):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0

            if self.checkpoint_dir:
                best_path = os.path.join(self.checkpoint_dir, f"{self.model_name}_best.pth")
                torch.save(model.state_dict(), best_path)
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
