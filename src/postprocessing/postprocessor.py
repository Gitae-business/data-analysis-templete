import numpy as np

class PostProcessor:
    def __init__(self, method="sigmoid", clip_range=(0, 1)):
        self.method = method
        self.clip_range = clip_range

    def apply(self, preds):
        preds = self._apply_method(preds)
        preds = self._apply_clipping(preds)
        return preds

    def _apply_method(self, preds):
        if self.method == "sigmoid":
            return 1 / (1 + np.exp(-preds))
        elif self.method == "softmax":
            exp_preds = np.exp(preds - np.max(preds, axis=1, keepdims=True))
            return exp_preds / np.sum(exp_preds, axis=1, keepdims=True)
        elif self.method == "zscore":
            return (preds - np.mean(preds)) / np.std(preds)
        elif self.method == "none":
            return preds
        else:
            raise ValueError(f"Unsupported postprocessing method: {self.method}")

    def _apply_clipping(self, preds):
        if self.clip_range is not None:
            low, high = self.clip_range
            preds = np.clip(preds, low, high)
        return preds
