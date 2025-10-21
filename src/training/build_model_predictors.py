from torch import optim
from config import config
from src.training.trainer import Trainer
from src.inference.predictor import Predictor
from src.inference.fold_average import FoldAveragePredictor

def build_model_predictors(model_name, train_df, criterion, device, checkpoint_dir, n_splits=5):
    trainer = Trainer(
        model_name=model_name,
        criterion=criterion,
        optimizer_cls=optim.AdamW,
        optimizer_params={
            'lr': config.LEARNING_RATE,
            'weight_decay': config.WEIGHT_DECAY
        },
        batch_size=config.BATCH_SIZE,
        device=device,
        checkpoint_dir=checkpoint_dir
    )

    models, preprocessors, _ = trainer.train(train_df, n_splits=n_splits)

    predictors = [Predictor(model=m, preprocessor=p, model_name=model_name, fold=i)
                  for i, (m, p) in enumerate(zip(models, preprocessors))]

    fold_avg_predictors = [FoldAveragePredictor(predictors[i:i+n_splits])
                           for i in range(0, len(predictors), n_splits)]

    return fold_avg_predictors
