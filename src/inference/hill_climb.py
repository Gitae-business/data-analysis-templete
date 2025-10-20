import numpy as np

class HillClimbEnsembler:
    def __init__(
        self, 
        predictors, 
        criterion,
        y_valid, 
        X_valid, 
        max_iter=1000, 
        step=0.05,
    ):
        self.predictors = predictors
        self.y_valid = y_valid
        self.X_valid = X_valid
        self.max_iter = max_iter
        self.step = step
        self.criterion = criterion

    def ensemble(self):
        preds_list = [p.predict(self.X_valid).flatten() for p in self.predictors]
        weights = np.ones(len(preds_list)) / len(preds_list)
        best_score = self._evaluate(weights, preds_list)

        print(f"Initial loss: {best_score:.6f}")

        for i in range(self.max_iter):
            idx = np.random.randint(0, len(weights))
            new_weights = weights.copy()
            new_weights[idx] += np.random.choice([-self.step, self.step])
            new_weights = np.clip(new_weights, 0, 1)
            new_weights /= new_weights.sum()

            score = self._evaluate(new_weights, preds_list)
            if score < best_score:
                best_score = score
                weights = new_weights
                print(f"Iteration {i}: Improved loss={best_score:.6f}")

        print(f"Final ensemble loss={best_score:.6f}")
        return weights

    def _evaluate(self, weights, preds_list):
        blended = np.average(preds_list, axis=0, weights=weights)
        return self.criterion(self.y_valid, blended)
