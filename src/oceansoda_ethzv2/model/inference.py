"""
perform inference on a single time step (data still in table format)
"""

import pandas as pd
import torch


def predict(model: torch.nn.Module, data: pd.DataFrame, device: str) -> pd.DataFrame:
    """
    Perform inference on a single time step.

    Args:
        model: The trained model.
        data: DataFrame containing the input data.
        device: Device to use for inference (e.g., 'cuda' or 'cpu').

    Returns:
        DataFrame with predictions.
    """
    model.eval()
    with torch.no_grad():
        # Convert DataFrame to tensor
        inputs = torch.tensor(data.values, dtype=torch.float32).to(device)

        # Perform inference
        outputs = model(inputs)

        # Convert outputs back to DataFrame
        predictions = pd.DataFrame(
            outputs.cpu().numpy(), columns=data.columns, index=data.index
        )

    return predictions
