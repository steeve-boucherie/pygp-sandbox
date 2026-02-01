"""Metrics for model fitting assessment."""
import logging
from typing import Any, Callable, List, Literal, Mapping, Union

import numpy as np

import pandas as pd
from pandas import DataFrame, Series

from prettytable import PrettyTable


# PACKAGE IMPORTS
from gp_sand.utils import rename_index, to_list


# LOGGER
logger = logging.getLogger(__name__)


# SCORE METHODS
def bias(
    y_pred: np.ndarray,
    y_actual: np.ndarray,
    axis: int = None,
    **kwargs: Mapping[str, Any],
) -> float | np.ndarray:
    """
    Given arrays of predicted and actual values, \
        compute corresponding bias.

    Notes
    -----
    bias = mean(y_pred - y_actual)

    Parameters
    ----------
    y_pred: np.ndarray
        Array of predicted values.
    y_actual: np.ndarray
        Array of actual values.
    axis: int
        Axis to use for aggregation.

    Returns
    -------
        float
    """
    # Check inputs
    # is1d(y_pred)
    # is1d(y_actual)

    return np.nanmean(y_pred - y_actual, axis=axis, **kwargs)


def bias_perc(
    y_pred: np.ndarray,
    y_actual: np.ndarray,
    axis: int = None
) -> float | np.ndarray:
    """
    Given arrays of predicted and actual values, \
        compute corresponding bias normalized by the actual mean.

    Notes
    -----
    bias_perc = mean(y_pred - y_actual) / mean(y_actual)

    Parameters
    ----------
    y_pred: np.ndarray
        Array of predicted values.
    y_actual: np.ndarray
        Array of actual values.
    axis: int
        Axis to use for aggregation.

    Returns
    -------
        float
    """
    # Check inputs
    # is1d(y_pred)
    # is1d(y_actual)
    num = np.nanmean(y_pred - y_actual, axis=axis)
    den = np.nanmean(y_actual, axis)

    return 100 * num / den


def mae(
    y_pred: np.ndarray,
    y_actual: np.ndarray,
    axis: int = None,
    center: bool = False,
) -> float | np.ndarray:
    """
    Given arrays of predicted and actual values, \
        compute corresponding Mean Absolute Error (MAE).

    Notes
    -----
    mae = mean(abs(y_pred - y_actual))

    Parameters
    ----------
    y_pred: np.ndarray
        Array of predicted values.
    y_actual: np.ndarray
        Array of actual values.
    center: bool
        An option to center the data (remove the bias). \
        Default is False.

    Returns
    -------
        float
    """
    # # Check inputs
    # is1d(y_pred)
    # is1d(y_actual)
    err = y_pred - y_actual
    offset = [0, bias(y_pred, y_actual, axis, keepdims=True)][center]

    return np.nanmean(np.abs(err - offset), axis)


def mae_perc(
    y_pred: np.ndarray,
    y_actual: np.ndarray,
    axis: int = None,
    center: bool = False,
) -> float | np.ndarray:
    """
    Given arrays of predicted and actual values, \
        compute corresponding Mean Absolute Error (MAE) normalized by the actual mean.

    Notes
    -----
    mae = mean(abs(y_pred - y_actual))

    Parameters
    ----------
    y_pred: np.ndarray
        Array of predicted values.
    y_actual: np.ndarray
        Array of actual values.
    center: bool
        An option to center the data (remove the bias). \
        Default is False

    Returns
    -------
        float
    """
    # # Check inputs
    # is1d(y_pred)
    # is1d(y_actual)
    err = y_pred - y_actual
    offset = [0, bias(y_pred, y_actual, axis, keepdims=True)][center]
    num = np.nanmean(np.abs(err - offset), axis)

    den = np.nanmean(y_actual, axis)

    return 100 * num / den


def mape(
    y_pred: np.ndarray,
    y_actual: np.ndarray,
    axis: int = None
) -> float | np.ndarray:
    """
    Given arrays of predicted and actual values, \
        compute corresponding Mean Absolute Percentage Error (MAPE).

    Notes
    -----
    mape = mean(abs(y_pred / y_actual - 1))

    Parameters
    ----------
    y_pred: np.ndarray
        Array of predicted values.
    y_actual: np.ndarray
        Array of actual values.

    Returns
    -------
        float
    """
    # # Check inputs
    # is1d(y_pred)
    # is1d(y_actual)

    return 100 * np.nanmean(np.abs(y_pred / y_actual - 1), axis=axis)


def mpe(
    y_pred: np.ndarray,
    y_actual: np.ndarray,
    axis: int = None
) -> float | np.ndarray:
    """
    Given arrays of predicted and actual values, \
        compute corresponding Mean Percentage Error (MPE).

    Notes
    -----
    mpe = mean(y_pred / y_actual - 1)

    Parameters
    ----------
    y_pred: np.ndarray
        Array of predicted values.
    y_actual: np.ndarray
        Array of actual values.

    Returns
    -------
        float
    """
    # # Check inputs
    # is1d(y_pred)
    # is1d(y_actual)

    return 100 * np.nanmean((y_pred / y_actual - 1), axis=axis)


def rmse(
    y_pred: np.ndarray,
    y_actual: np.ndarray,
    axis: int = None,
    center: bool = False,
) -> float | np.ndarray:
    """
    Given arrays of predicted and actual values, \
        compute corresponding Root Mean Square Error (RMSE).

    Notes
    -----
    rmse = sqrt(mean((y_pred - y_actual)**2))

    Parameters
    ----------
    y_pred: np.ndarray
        Array of predicted values.
    y_actual: np.ndarray
        Array of actual values.
    center: bool
        An option to center the data (remove the bias). \
        Default is False.

    Returns
    -------
        float
    """
    # # Check inputs
    # is1d(y_pred)
    # is1d(y_actual)
    err = y_pred - y_actual
    offset = [0, bias(y_pred, y_actual, axis, keepdims=True)][center]

    return np.nanmean((err - offset)**2, axis)**.5


def cov(
    y_pred: np.ndarray,
    y_actual: np.ndarray,
    axis: int = None,
    center: bool = False,
    unit: Literal['fraction', 'percent'] = 'percent',
) -> float | np.ndarray:
    """
    Given arrays of predicted and actual values, \
        compute corresponding Coefficient of variations (CoV).

    Notes
    -----
    >>> cov = sqrt(mean((y_pred - y_actual)**2)) / mean(y_actual)

    Parameters
    ----------
    y_pred: np.ndarray
        Array of predicted values.
    y_actual: np.ndarray
        Array of actual values.
    axis: int | None
        (Optional) Axis along which to compute the aggregation \
        aggrgate all axes if None is passed.
    center: bool
        An option to center the data (remove the bias). \
        Default is False.
    unit: 'fraction' | 'percent'
        The unit in which the result should be displayed.

    Returns
    -------
        float
    """
    # Check inputs
    if unit not in ['percent', 'fraction']:
        msg = 'Invalid valid values for parameter "unit".\nMust be one of: ' \
              f'{['percent', 'fraction']} but received "{unit}".\n' \
              'Please check your inputs.'
        logger.error(msg)
        raise ValueError(msg)

    fact = {'percent': 100, 'fraction': 1}[unit]
    err = y_pred - y_actual
    offset = [0, bias(y_pred, y_actual, axis, keepdims=True)][center]
    num = np.nanmean((err - offset)**2, axis)**.5
    den = np.nanmean(y_actual, axis)

    return fact * num / den


def nrmse(
    y_pred: np.ndarray,
    y_actual: np.ndarray,
    axis: int = None,
    center: bool = False,
    unit: Literal['fraction', 'percent'] = 'percent',
) -> float | np.ndarray:
    """
    Given arrays of predicted and actual values, \
        compute the corresponding Normalize Root Mean Square Error (NRMSE).

    Notes
    -----
    >>> nrmse = sqrt(mean((y_pred - y_actual)**2)) / std(y_actual)

    Parameters
    ----------
    y_pred: np.ndarray
        Array of predicted values.
    y_actual: np.ndarray
        Array of actual values.
    axis: int | None
        (Optional) Axis along which to compute the aggregation \
        aggrgate all axes if None is passed.
    center: bool
        An option to center the data (remove the bias). \
        Default is False.
    unit: 'fraction' | 'percent'
        The unit in which the result should be displayed.

    Returns
    -------
        float
    """
    # Check inputs
    if unit not in ['percent', 'fraction']:
        msg = 'Invalid valid values for parameter "unit".\nMust be one of: ' \
              f'{['percent', 'fraction']} but received "{unit}".\n' \
              'Please check your inputs.'
        logger.error(msg)
        raise ValueError(msg)

    fact = {'percent': 100, 'fraction': 1}[unit]
    err = y_pred - y_actual
    offset = [0, bias(y_pred, y_actual, axis, keepdims=True)][center]
    num = np.nanmean((err - offset)**2, axis)**.5
    den = np.nanstd(y_actual, axis)

    return fact * num / den


def rsquare(
    y_pred: np.ndarray,
    y_actual: np.ndarray,
    axis: int = None
) -> float | np.ndarray:
    """
    Given arrays of predicted and actual values, \
        compute corresponding R-square (R2).

    Notes
    -----
    r2 = 1 - mean(y_pred - y_actual)**2 / var(y_pred)**2

    Parameters
    ----------
    y_pred: np.ndarray
        Array of predicted values.
    y_actual: np.ndarray
        Array of actual values.

    Returns
    -------
        float
    """
    # # Check inputs
    # is1d(y_pred)
    # is1d(y_actual)

    # Get the terms
    ss_res = ((y_pred - y_actual)**2).sum(axis=axis)
    ss_tot = ((y_actual - y_actual.mean())**2).sum(axis=axis)

    return 1 - ss_res / ss_tot


# DISPLAY

def compute_scores(
    pred: np.ndarray | Series | DataFrame,
    actual: np.ndarray | Series | DataFrame,
    methods: Union[
        Callable[[np.ndarray], float],
        List[Callable[[np.ndarray], float]]
    ]
) -> DataFrame:
    """
    Given arrays of predicted and actual values, \
        compute corresponding scores and return results in a Series object.

    Parameters
    ----------
    pred: np.ndarray
        Array of predicted values.
    actual: np.ndarray
        Array of actual values.

    Returns
    -------
        Series
    """
    # Check inputs
    if isinstance(pred, np.ndarray):
        pred = Series(pred, name='pred')
    if isinstance(actual, np.ndarray):
        actual = Series(actual, name='actual')
    if type(pred) is not type(actual):
        msg = 'Input must be of the same type.'
        logger.error(msg)
        raise ValueError(msg)

    # Force input type
    methods = to_list(methods)

    if isinstance(pred, Series):
        columns = [pred.name]

    else:
        columns = pred.columns

    scores = DataFrame(
        data=[fun(pred, actual, axis=0) for fun in methods],
        columns=columns,
        index=pd.Index(
            data=[fun.__name__ for fun in methods],
            name='metric',
        )
    )

    scores = rename_index(scores, 'target')
    # scores.

    return scores.T


def display_scores(scores: DataFrame) -> None:
    """
    Given the prediciton scores, display them in the logger.

    Parameters
    ----------
    scores: Series
        Input scores to be displayed.
    """
    table = PrettyTable()
    table.field_names = [scores.index.name] + list(scores.columns)

    # iterate on index
    for ind in scores.index:
        row = scores.loc[ind, :]
        table.add_row([ind] + [f'{sc:.3f}' for sc in row])

    logger.info(f'Prediction scores:\n{table}')
