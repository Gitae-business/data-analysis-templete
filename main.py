import numpy as np
import torch.nn as nn
from config import config
from src.loader.loader import load_data
from sklearn.model_selection import train_test_split
from src.analysis.eda import analyze_data
from src.training.build_model_predictors import build_model_predictors
from src.inference.hill_climb import HillClimbEnsembler

def main():
    train_df, test_df = load_data()
    # analyze_data(train_df)

    train_df_split, val_df_split = train_test_split(train_df, test_size=0.2, random_state=42)
    y_val = val_df_split[config.TARGET]
    X_val = val_df_split.drop(config.TARGET, axis=1)
    
    criterion = nn.HuberLoss()

    model_list = ["MLP", "Linear"]
    all_model_predictors = []

    for model_name in model_list:
        predictors = build_model_predictors(
            model_name=model_name,
            train_df=train_df_split,
            criterion=criterion,
            device=config.DEVICE,
            checkpoint_dir=config.CHECKPOINT_DIR,
            n_splits=config.OOF_SPLIT
        )
        all_model_predictors += predictors

    # Ensemble
    ensembler = HillClimbEnsembler(
        predictors=all_model_predictors,
        criterion=criterion,
        y_valid=y_val,
        X_valid=X_val,
        max_iter=200,
        step=0.05
    )

    best_weights = ensembler.ensemble()
    print(f"Best ensemble weights: {best_weights}")

    test_preds_list = [p.predict(test_df).flatten() for p in all_model_predictors]
    test_preds = np.average(test_preds_list, axis=0, weights=best_weights)
    print(f"Test predictions shape: {test_preds.shape}")

if __name__ == '__main__':
    main()
