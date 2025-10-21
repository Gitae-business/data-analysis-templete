import torch
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
        self.criterion = criterion
        self.y_valid = y_valid
        self.X_valid = X_valid
        self.max_iter = max_iter
        self.step = step

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
        blended = np.zeros_like(preds_list[0], dtype=np.float32)
        for w, pred in zip(weights, preds_list):
            blended += float(w) * np.asarray(pred, dtype=np.float32)

        if hasattr(self.y_valid, "values"):
            y_np = np.asarray(self.y_valid.values, dtype=np.float32)
        else:
            y_np = np.asarray(self.y_valid, dtype=np.float32)

        if isinstance(self.criterion, torch.nn.Module):
            blended_t = torch.from_numpy(blended).float()
            y_t = torch.from_numpy(y_np).float()

            if blended_t.dim() == 1:
                blended_t = blended_t.view(-1)
            if y_t.dim() == 2 and y_t.shape[1] == 1:
                y_t = y_t.view(-1)

            loss = self.criterion(blended_t, y_t)
            return float(loss.item())
        else:
            return float(self.criterion(y_np, blended))
