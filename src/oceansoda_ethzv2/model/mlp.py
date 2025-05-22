import torch
from torch import nn


class TwoLayerMLP(nn.Module):
    def __init__(self, input_size: int, output_size: int, dropout_prob: float = 0.2):
        super(TwoLayerMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.activation1 = nn.ReLU()

        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.activation2 = nn.ReLU()

        self.fc_out = nn.Linear(32, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.dropout1(x)
        x = self.activation1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.dropout2(x)
        x = self.activation2(x)

        x = self.fc_out(x)
        return x


class EnsembleMLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_members: int = 5,
        dropout_prob: float = 0.2,
    ):
        super(EnsembleMLP, self).__init__()
        self.models = nn.ModuleList(
            [
                TwoLayerMLP(input_size, output_size, dropout_prob)
                for _ in range(num_members)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the ensemble of models, where each model receives i from dim0 in x

        Args:
            x: Input tensor of shape (batch_size, input_size)
        """
        outputs = [model(x) for model in self.models]
        # concatenate outputs into single tensor
        return torch.cat(outputs, dim=0)

    def to(self, device: str):
        super(EnsembleMLP, self).to(device)
        for model in self.models:
            model.to(device)

    def from_model_list(self, model_list: list):
        """Create an EnsembleMLP from a list of models."""

        self.models = nn.ModuleList(model_list)
        self.num_members = len(model_list)
        self.input_size = model_list[0].input_size
        self.output_size = model_list[0].output_size
        self.dropout_prob = model_list[0].dropout_prob
