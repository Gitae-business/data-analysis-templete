import os
import copy
import torch
import numpy as np
from tqdm import tqdm
from config import config
from sklearn.model_selection import KFold
from src.models.model_factory import get_model
from src.models.earlystop import EarlyStopping
from torch.utils.data import TensorDataset, DataLoader
from src.preprocessing.preprocessor import DataPreprocessor

class Trainer:
    def __init__(
        self,
        model_name="MLP",
        criterion=None,
        optimizer_cls=None,
        optimizer_params=None,
        batch_size=None,
        device=None,
        checkpoint_dir=None,
        early_stop_patience=5,
        n_splits=5,
        **model_kwargs
    ):
        self.device = device or config.DEVICE
        self.batch_size = batch_size or config.BATCH_SIZE

        self.preprocessor = DataPreprocessor()
        self.model_name = model_name
        self.model_kwargs = model_kwargs
        self.n_splits = n_splits

        self.checkpoint_dir = checkpoint_dir or os.path.join(config.MODEL_PATH, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.criterion = criterion
        self.optimizer_cls = optimizer_cls
        self.optimizer_params = optimizer_params or {}

        self.early_stop_patience = early_stop_patience

    def train(self, train_df, n_splits=5, n_epochs=None):
        n_epochs = n_epochs or int(config.EPOCHS)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        oof_preds = np.zeros(len(train_df))
        models = []
        preprocessors = []

        for fold, (train_idx, valid_idx) in enumerate(kf.split(train_df)):
            print(f"\n===== Fold {fold+1}/{n_splits} =====")
            train_fold = train_df.iloc[train_idx]
            valid_fold = train_df.iloc[valid_idx]

            preprocessor_fold = copy.deepcopy(self.preprocessor)
            X_train, y_train = preprocessor_fold.fit_transform(train_fold)
            X_valid, y_valid = preprocessor_fold.transform(valid_fold.drop(config.TARGET, axis=1)), valid_fold[config.TARGET]

            input_dim = X_train.shape[1]
            model_fold = get_model(input_dim, model_name=self.model_name, **self.model_kwargs).to(self.device)
            optimizer_fold = self.optimizer_cls(model_fold.parameters(), **self.optimizer_params)

            X_train_tensor = torch.FloatTensor(X_train).to(self.device)
            y_train_tensor = torch.FloatTensor(y_train.values).unsqueeze(1).to(self.device)
            X_valid_tensor = torch.FloatTensor(X_valid).to(self.device)
            y_valid_tensor = torch.FloatTensor(y_valid.values).unsqueeze(1).to(self.device)

            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

            early_stopper = EarlyStopping(
                patience=self.early_stop_patience,
                verbose=True,
                checkpoint_dir=self.checkpoint_dir,
                model_name=f"{config.MODEL_NAME}_fold{fold+1}"
            )

            for epoch in range(n_epochs):
                model_fold.train()
                epoch_loss = 0
                loop = tqdm(train_loader, desc=f"Fold {fold+1} Epoch {epoch+1}/{n_epochs}", leave=False)
                
                for batch_X, batch_y in loop:
                    optimizer_fold.zero_grad()
                    outputs = model_fold(batch_X)
                    loss = self.criterion(outputs, batch_y)
                    loss.backward()
                    optimizer_fold.step()
                    epoch_loss += loss.item()
                    loop.set_postfix({'Train Loss': f'{epoch_loss / (loop.n + 1):.4f}'})
                epoch_loss /= len(train_loader)

                model_fold.eval()
                with torch.no_grad():
                    outputs_val = model_fold(X_valid_tensor)
                    val_loss = self.criterion(outputs_val, y_valid_tensor).item()

                print(f"Epoch {epoch+1}/{n_epochs} | Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f}")

                early_stopper(val_loss, model_fold, epoch)
                if early_stopper.early_stop:
                    print(f"Early stopping triggered at epoch {epoch+1} of fold {fold+1}")
                    break

            best_model_path = os.path.join(self.checkpoint_dir, f"{config.MODEL_NAME}_fold{fold+1}_best.pth")
            if os.path.exists(best_model_path):
                model_fold.load_state_dict(torch.load(best_model_path))

            models.append(model_fold)
            preprocessors.append(preprocessor_fold)
            with torch.no_grad():
                oof_preds[valid_idx] = model_fold(X_valid_tensor).cpu().numpy().flatten()

        return models, preprocessors, oof_preds
