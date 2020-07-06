import numpy as np
from deeppavlov.core.common.metrics_registry import register_metric


@register_metric('mean_squared_error')
def mse(y_true, y_predicted):
    """
    Calculates mean squared error.
    Args:
        y_true: list of true probs
        y_predicted: list of predicted peobs
    Returns:
        F1 score
    """
    # raise Exception(str(y_true[:5]))
    for value in [y_true, y_predicted]:
        assert (np.isfinite(value).all())
    y_true_np = np.array(y_true)
    assert len(y_true_np.shape) == 2
    y_predicted_np = np.array(y_predicted)
    assert len(y_predicted_np.shape) == 2
    err = ((y_true_np - y_predicted_np) ** 2).sum() ** 0.5
    return err
