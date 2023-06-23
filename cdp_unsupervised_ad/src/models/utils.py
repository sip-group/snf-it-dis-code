import os
import torch

DEFAULT_T2X_MODEL_PATH = "t2x.pt"
DEFAULT_T2XA_MODEL_PATH = "t2xa.pt"
DEFAULT_X2T_MODEL_PATH = "x2t.pt"
DEFAULT_X2TA_MODEL_PATH = "x2ta.pt"

MODE_TO_PATHS = {
    "t2x": [DEFAULT_T2X_MODEL_PATH],
    "t2xa": [DEFAULT_T2XA_MODEL_PATH],
    "x2t": [DEFAULT_X2T_MODEL_PATH],
    "x2ta": [DEFAULT_X2TA_MODEL_PATH],
    "both": [DEFAULT_T2X_MODEL_PATH, DEFAULT_X2T_MODEL_PATH],
    "both_a": [DEFAULT_T2XA_MODEL_PATH, DEFAULT_X2TA_MODEL_PATH]
}


def normalize(tensor):
    """Returns a tensor 'normalized' in range [0, 1]"""
    result = tensor - torch.min(tensor)
    result = result / torch.max(result) if torch.max(result) > 0 else torch.ones(result.shape)
    return result


def forward(mode, models, t, x):
    """Method which returns the loss for the given mode using the given models, templates and originals"""
    if mode == "t2x":
        x_hat = models[0](t)
        return torch.mean((x_hat - x) ** 2), x_hat
    elif mode == "t2xa":
        x_hat, c = models[0](t).chunk(2, 1)
        diff = (x - x_hat)
        return torch.mean(diff ** 2) + torch.mean((normalize(1 - torch.abs(diff)) - c) ** 2), x_hat, c
    elif mode == "x2t":
        t_hat = models[0](x)
        return torch.mean((t_hat - t) ** 2), t_hat
    elif mode == "x2ta":
        t_hat, c = models[0](x).chunk(2, 1)
        diff = (t - t_hat)
        return torch.mean(diff ** 2) + torch.mean((normalize(1 - torch.abs(diff)) - c) ** 2), t_hat, c
    elif mode == "both":
        t2x_model, x2t_model = models[0], models[1]
        x_hat = t2x_model(t)
        t_hat = x2t_model(x)

        l_standard = (torch.mean((x_hat - x) ** 2) + torch.mean((t_hat - t) ** 2)) / 2
        return l_standard, x_hat, t_hat
    elif mode == "both_a":
        t2xa_model, x2ta_model = models[0], models[1]
        x_hat, cx = t2xa_model(t).chunk(2, 1)
        t_hat, ct = x2ta_model(x).chunk(2, 1)

        l_cyc = torch.mean((x2ta_model(x_hat).chunk(2, 1)[0] - t) ** 2) + \
                torch.mean((t2xa_model(t_hat).chunk(2, 1)[0] - x) ** 2)

        x_diff, t_diff = x_hat - x, t_hat - t
        l_standard = torch.mean(x_diff ** 2) + torch.mean((normalize(1 - torch.abs(x_diff)) - cx) ** 2) + \
                     torch.mean(t_diff ** 2) + torch.mean((normalize(1 - torch.abs(t_diff)) - ct) ** 2)

        return l_cyc + l_standard, x_hat, cx, t_hat, ct
    else:
        raise KeyError(f"Unknown mode {mode}!")


def store_models(mode, models, dest):
    """Stores the models in the result directory."""
    for model, name in zip(models, MODE_TO_PATHS[mode]):
        torch.save(model, os.path.join(dest, name))


def load_models(mode, models_dir, device):
    """Loads the trained models depending on the mode onto the device."""
    models = []
    for name in MODE_TO_PATHS[mode]:
        models.append(torch.load(os.path.join(models_dir, name), map_location=device))
    return models
