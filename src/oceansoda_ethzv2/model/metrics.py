import xarray as xr
from sklearn import metrics as sklearn_metrics


def compute_metric(y_true: xr.DataArray, y_pred: xr.DataArray, metric_name: str):
    """
    Compute a metric for the given data with masking applied

    Parameters
    ----------
    y_true: xr.DataArray
        The true values
    y_pred: xr.DataArray
        The predicted values
    metric_name: str
        The name of the metric to compute. Options are 'r2', 'rmse', 'mae', 'mse', 'mape'
    """

    # check arrays are same size
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")

    mask = (y_true.isnull() | y_pred.isnull()).values
    y_true_masked = y_true.values[~mask]
    y_pred_masked = y_pred.values[~mask]

    func = getattr(sklearn_metrics, metric_name)
    result = func(y_true_masked, y_pred_masked)

    return result


def r2_score(y_true: xr.DataArray, y_pred: xr.DataArray):
    return compute_metric(y_true, y_pred, "r2_score")


def rmse(y_true: xr.DataArray, y_pred: xr.DataArray):
    return compute_metric(y_true, y_pred, "rmse")


def mae(y_true: xr.DataArray, y_pred: xr.DataArray):
    return compute_metric(y_true, y_pred, "mae")
