"""A bunch of utility stuff"""
import logging
import warnings
from typing import Any, List, Literal, Mapping

import numpy as np

from pandas import DataFrame

from sklearn.utils import resample

import torch
from torch import Tensor


# LOGGER
logger = logging.getLogger(__name__)


# FORMATTING
def to_list(x: Any | List[Any]) -> List[Any]:
    """
    Convert the input variable to a list.

    Notes
    -----
    Without effect if x is already of list type.

    Parameters
    ----------
    x: Any | List[Any]
        Input variable
    allowed: List[Any]
        The list of allowed values.

    Raises
    ------
        ValueError
    """
    return [[x], x][isinstance(x, list)]


def is_allowed(val: Any, allowed: List[Any]) -> None:
    """
    Test if the input value is one of the allowed values, \
        raise ValueError if otherwise.

    Parameters
    ----------
    val: Any
        The value to be tested.
    allowed: List[Any]
        The list of allowed values.

    Raises
    ------
        ValueError
    """
    if val not in allowed:
        msg = f'Invalid value "{val}", must be one of "{allowed}". ' \
              'Please check your inputs.'
        logger.error(msg)
        raise ValueError(msg)


def rename_index(
    df: DataFrame,
    name: str,
    axis: Literal['index', 'columns'] = 'columns'
) -> DataFrame:
    """
    Rename the index or columns of the input single-level indexed dataframe.

    Parameters
    ----------
    df: DataFrame
        Input dataframe.
    name: str
        New name of the index/column.
    axis: 'index' | 'column'
        Whether the index or columns should be renamed.

    Returns
    -------
        DataFrame
    """
    # Check inputs
    is_allowed(axis, ['index', 'columns'])

    # Rename
    if axis == 'index':
        df.index = df.index.rename(name)

    else:
        df.columns = df.columns.rename(name)

    return df


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


def to_numpy(
    x: np.ndarray | Tensor,
) -> np.ndarray:
    """Convert the input to numpy if not already"""
    if isinstance(x, Tensor):
        x = x.numpy()

    return x


def to_tensor(
    x: np.ndarray | Tensor,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Convert the input to Tensor if not already"""
    if isinstance(x, np.ndarray):
        x = torch.tensor(x).to(dtype)

    return x


# TRAINING GPs
def get_inductions_points(
    x_feat: np.ndarray | Tensor,
    n_points: int = 150,
) -> Tensor:
    """
    Given the input features, compute the initial locations \
        of the inductions points for sparse GP training.

    Parameters
    ----------
    x_feat: np.ndarray | Tensor
        Input array of features.
    n_points: int
        The number of induction points to generate for each variables

    Returns
    -------
        Tensor
    """
    x_feat = to_numpy(x_feat)
    ind_points = np.percentile(
        x_feat,
        np.linspace(0, 100, n_points),
        axis=0
    )

    x_feat = np.atleast_2d(x_feat)
    np.random.random((n_points, x_feat.shape[1]))

    return torch.tensor(ind_points).to(torch.float32)


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
