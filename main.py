import numpy as np
import torch.nn as nn
import torch.optim as optim
from config import config
from src.loader.loader import load_data
from sklearn.model_selection import train_test_split
from src.analysis.eda import analyze_data
from src.models.model_factory import get_model
from src.training.trainer import Trainer
from src.inference.predictor import Predictor
from src.inference.hill_climb import HillClimbEnsembler

def main():
    train_df, test_df = load_data()
    # analyze_data(train_df)

    train_df_split, val_df_split = train_test_split(train_df, test_size=0.2, random_state=42)
    y_val = val_df_split[config.TARGET]
    X_val = val_df_split.drop(config.TARGET, axis=1)
    
    criterion = nn.HuberLoss()

    # MLP Model
    trainer_mlp = Trainer(
        model_name="MLP",
        criterion=criterion,
        optimizer_cls=optim.AdamW,
        optimizer_params={
            'lr': config.LEARNING_RATE, 
            'weight_decay': config.WEIGHT_DECAY
        },
        batch_size=config.BATCH_SIZE,
        device=config.DEVICE,
        checkpoint_dir=config.CHECKPOINT_DIR
    )
    mlp_models, mlp_preprocessors, mlp_oof = trainer_mlp.train(train_df_split, n_splits=5)
    predictors_mlp = [Predictor(model=m, preprocessor=p) for m, p in zip(mlp_models, mlp_preprocessors)]

    # Linear Model
    trainer_lin = Trainer(
        model_name="Linear",
        criterion=criterion,
        optimizer_cls=optim.AdamW,
        optimizer_params={
            'lr': config.LEARNING_RATE, 
            'weight_decay': config.WEIGHT_DECAY
        },
        batch_size=config.BATCH_SIZE,
        device=config.DEVICE,
        checkpoint_dir=config.CHECKPOINT_DIR
    )
    lin_models, lin_preprocessors, lin_oof = trainer_lin.train(train_df_split, n_splits=5)
    predictors_lin = [Predictor(model=m, preprocessor=p) for m, p in zip(lin_models, lin_preprocessors)]

    # Ensemble
    all_predictors = predictors_mlp + predictors_lin
    ensembler = HillClimbEnsembler(
        predictors=all_predictors,
        criterion=criterion,
        y_valid=y_val,
        X_valid=X_val,
        max_iter=200,
        step=0.05
    )
    best_weights = ensembler.ensemble()
    print(f"Best ensemble weights: {best_weights}")

    test_preds_list = [p.predict(test_df).flatten() for p in all_predictors]
    test_preds = np.average(test_preds_list, axis=0, weights=best_weights)
    print(f"Test predictions shape: {test_preds.shape}")

if __name__ == '__main__':
    main()
