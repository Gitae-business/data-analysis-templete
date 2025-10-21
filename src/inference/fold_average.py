import numpy as np

class FoldAveragePredictor:
    def __init__(self, predictors_folds):
        self.predictors_folds = predictors_folds

    def predict(self, X):
        fold_preds = [p.predict(X).flatten() for p in self.predictors_folds]
        return np.mean(fold_preds, axis=0)
