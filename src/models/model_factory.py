import torch.nn as nn

class MLPModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)

class SimpleLinear(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)

def get_model(input_dim, model_name="MLP", **kwargs):
    if model_name == "MLP":
        return MLPModel(input_dim, **kwargs)
    elif model_name == "Linear":
        return SimpleLinear(input_dim, **kwargs)
    else:
        raise ValueError(f"Unknown model_name: {model_name}. Available: ['MLP', 'Linear']")
