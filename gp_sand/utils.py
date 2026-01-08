"""A bunch of utility stuff"""
import logging
import warnings
from typing import Any, Literal, Mapping

import numpy as np

from pandas import DataFrame

from sklearn.utils import resample


# LOGGER
logger = logging.getLogger(__name__)


# PREPROCESSING
def resample_df(
    df: DataFrame,
    n_samples: int = 5000,
    strat_col: str | None = None,
    reset_index: bool = True,
    **rsmp_kw: Mapping[str, Any],
) -> DataFrame:
    """
    Given the dataframe resample to the selected number of samples.

    Parameters
    ----------
    df: DataFrame
        Input dataframe
    n_samples: int
        Number of samples
    start_col: str | None
        (Optional) column from the input dataframe to use \
        for splitting data in a stratify fashion.
    reset_index:
        An option for whether to reset the index of the output dataframe.
    rsmp_kw: Mapping[str, Any]
        Optional argument to be passed to the sklearn.resample method.

    Returns
    -------
        DataFrame
    """
    def _get_defaults() -> Mapping[str, Any]:
        """Get defaults"""
        params = {'replace': False}
        return params

    rsmp_kw = _get_defaults() | rsmp_kw
    rsmp_kw['n_samples'] = n_samples
    if strat_col:
        # TODO: Make validator
        rsmp_kw['stratify'] = df[strat_col]

    try:
        out = resample(df, **rsmp_kw)

    except ValueError:
        msg = 'Resampling failed. This is likely due to an insuficient number rows' \
              f'of the input dataframe ({len(df)}) compared to the selected number' \
              f'of samples ({n_samples}) when replace option is set to False.\n' \
              'Returning the entire dataset.'
        logger.warning(msg)
        warnings.warn(msg)
        out = df

    if reset_index:
        out = out.reset_index(drop=True)

    return out


# METRICS
def compute_rmse(
    pred: np.ndarray,
    actual: np.ndarray,
    axis: int | None = None,
) -> float | np.ndarray:
    """
    Given the arrays of predcited and actual values, \
        compute the corresponding Root Mean Square Error (RMSE).

    Parameters
    ----------
    pred: np.ndarrray
        Array of predicted values.
    actual: np.ndarrray
        Array of actual values.
    axis: int | None
        (Optional) Index of the axes to be aggregated. Aggregate \
        all axes if None is passed.

    Returns
    -------
        float | np.ndarray
    """
    return np.sqrt(np.power(pred - actual, 2).mean(axis))


def compute_rmse_norm(
    pred: np.ndarray,
    actual: np.ndarray,
    axis: int | None = None,
    res_type: Literal['fraction', 'percent'] = 'percent',
) -> float | np.ndarray:
    """
    Given the arrays of predcited and actual values, compute the \
        corresponding normalized RMSE.

    Notes
    -----
    Normalize RMSE result by the standard deviation of the actual values.

    Parameters
    ----------
    pred: np.ndarrray
        Array of predicted values.
    actual: np.ndarrray
        Array of actual values.
    axis: int | None
        (Optional) Index of the axes to be aggregated. Aggregate \
        all axes if None is passed.

    Returns
    -------
        float | np.ndarray
    """
    allowed = ['fraction', 'percent']
    if res_type not in allowed:
        msg = f'Invalid selected result type "{res_type}".\n' \
              f'Value must be one of: {allowed}.\nPlease check your inputs.'
        logger.error(msg)
        raise ValueError(msg)

    rmse = compute_rmse(pred, actual, axis)
    std = actual.std(axis)
    fact = {'fraction': 1, 'percent': 100}[res_type]

    return fact * rmse / std
