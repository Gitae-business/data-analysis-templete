import os
import torch
import numpy as np
import pandas as pd
from config import config
from torch.utils.data import TensorDataset, DataLoader
from src.preprocessing.preprocessor import DataPreprocessor
from src.postprocessing.postprocessor import PostProcessor

class Predictor:
    def __init__(self, model, preprocessor: DataPreprocessor, model_name="MLP", fold=0):
        self.device = config.DEVICE
        self.model = model.to(self.device)
        self.model_name = model_name
        self.fold = fold

        self.model_path = os.path.join(config.CHECKPOINT_DIR, model_name, f"{model_name}_fold_{fold}.pth")

        self.preprocessor = preprocessor
        self.postprocessor = PostProcessor()

        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
        else:
            print(f"Warning: checkpoint not found at {self.model_path}")

        self.model.eval()

    def predict(self, test_df, batch_size=None, apply_postprocess=True):
        batch_size = batch_size or config.BATCH_SIZE

        X_test = self.preprocessor.transform(test_df)
        if hasattr(X_test, "toarray"):
            X_test_tensor = torch.FloatTensor(X_test.toarray()).to(self.device)
        else:
            X_test_tensor = torch.FloatTensor(X_test).to(self.device)

        dataset = TensorDataset(X_test_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        preds = []
        with torch.no_grad():
            for (batch_X,) in loader:
                outputs = self.model(batch_X.to(self.device))
                preds.extend(outputs.cpu().numpy())

        preds = np.array(preds)

        if apply_postprocess:
            preds = self.postprocessor.apply(preds)

        return preds

    def save_submission(self, preds):
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        save_path = os.path.join(config.OUTPUT_DIR, "submission.csv")

        try:
            submission_df = pd.read_csv(os.path.join(config.DATA_DIR, config.SUBMISSION))
            submission_df.iloc[:, 1] = preds.flatten()
        except FileNotFoundError:
            print("Submission file not found. Creating a temporary one.")
            submission_df = pd.DataFrame({"ID": range(len(preds)), "target": preds.flatten()})

        submission_df.to_csv(save_path, index=False)
        print(f"Submission file saved to: {save_path}")

        return save_path
