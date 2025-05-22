import torch


def train_epoch(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str = "cpu",
) -> float:
    """
    Train the model for one epoch.

    Args:
        model: The model to train.
        train_loader: DataLoader for the training data.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to use for training (e.g., 'cuda' or 'cpu').
    """
    model.train()
    running_loss = 0.0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    return epoch_loss
