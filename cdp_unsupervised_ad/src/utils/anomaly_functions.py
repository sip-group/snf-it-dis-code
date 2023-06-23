import torch


def keep_top_percent(confidence: torch.Tensor, percent: float, binarize=False):
    """Given a confidence tensor, masks (to zero) values that are below a percentage in the 0-th dimension"""
    assert 0 < percent < 1, f"Error: Percentage must be between 0 and 1 (excluded), but got {percent}"

    result = []
    for sample_i in confidence:
        t = torch.quantile(sample_i, percent)
        mask = sample_i > t

        modified = torch.clone(sample_i)
        modified[torch.logical_not(mask)] = 0

        if binarize:
            modified[mask] = 1

        result.append(modified)
    return torch.stack(result)


def get_anomaly_score(mode, models, t, y):
    """Given the mode, pre-trained model(s), templates and test printed CDPs, returns the anomaly scores per each sample
    based on the provided mode."""
    if mode == "t2x":
        t2x_model = models[0]
        x_hat = t2x_model(t)
        c = keep_top_percent(1 - torch.abs(x_hat - t), 0.85)
        anomaly_map = (c * (x_hat - y) ** 2)
        return torch.sum(anomaly_map, dim=[1, 2, 3]).detach().cpu().numpy()
    if mode == "x2t":
        return binarize_mse_af(t, models[0](y))
    if mode == "t2xa":
        y_hat, confidence = models[0](t).chunk(2, 1)
        return confidence_mse_af(y, y_hat, confidence)
    if mode == "x2ta":
        t_hat, confidence = models[0](y).chunk(2, 1)
        return confidence_mse_af(t, t_hat, confidence)
    if mode == "both":
        t2xa_model, x2ta_model = models[0], models[1]
        t2x_score = get_anomaly_score(Mode.MODE_T2X, [t2xa_model], t, y)
        x2t_score = get_anomaly_score(Mode.MODE_X2T, [x2ta_model], t, y)
        return (t2x_score ** 2 + x2t_score ** 2) ** 0.5
    if mode == "both_a":
        t2xa_model, x2ta_model = models[0], models[1]
        t2xa_score = get_anomaly_score(Mode.MODE_T2XA, [t2xa_model], t, y)
        x2ta_score = get_anomaly_score(Mode.MODE_X2TA, [x2ta_model], t, y)
        return (t2xa_score ** 2 + x2ta_score ** 2) ** 0.5


def mse_af(actual, estimate):
    """Returns the batch-wise MSE"""
    return torch.mean((estimate - actual) ** 2, dim=[1, 2, 3]).detach().cpu().numpy()


def binarize_mse_af(actual, estimate, threshold=.5):
    """Returns the batch-wise MSE by first applying a binarization on the estimate"""
    estimate[estimate < threshold] = 0
    estimate[estimate >= threshold] = 1
    return mse_af(actual, estimate)


def confidence_mse_af(actual, estimate, confidence):
    """Returns the batch-wise MSE weighted by a confidence tensor"""
    return torch.mean(confidence * (estimate - actual) ** 2, dim=[1, 2, 3]).detach().cpu().numpy()


def binarize_confidence_mse_af(actual, estimate, confidence, threshold=.5):
    """Returns the batch-wise MSE using a binarizing threshold and weighting by a confidence tensor"""
    estimate[estimate < threshold] = 0
    estimate[estimate >= threshold] = 1
    return torch.mean(confidence * (estimate - actual) ** 2, dim=[1, 2, 3]).detach().cpu().numpy()
