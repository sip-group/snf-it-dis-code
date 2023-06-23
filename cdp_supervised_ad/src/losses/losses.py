import torch
from torch import nn


def get_classification_loss(device=torch.device("cpu")):
    """Simple Binary Cross-entropy loss for binary classification (0-1)."""
    sigmoid = nn.Sigmoid()
    bce = nn.BCELoss()

    def classification_loss(t_lv, x_lvs, f_lvs):
        n_lvs = len(x_lvs) + len(f_lvs)
        batch_size = len(x_lvs[0])

        loss = 0.0
        for x_lv in x_lvs:
            y_hat = sigmoid(x_lv).flatten()
            y = torch.ones(batch_size).to(device)
            loss += bce(y_hat, y) / len(x_lvs)

        for f_lv in f_lvs:
            y_hat = sigmoid(f_lv).flatten()
            y = torch.zeros(batch_size).to(device)
            loss += bce(y_hat, y) / len(f_lvs)

        return loss

    return classification_loss
