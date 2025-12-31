"""Class and method for simple univariate GP without parameter optim."""
import logging
from typing import Callable, Tuple

import numpy as np

from scipy import stats

from sklearn.exceptions import NotFittedError


# LOGGER
logger = logging.getLogger(__name__)


# DEFAULT
RANDOM_SEED: int = 28091993


# UTILS
def _rng(random_seed: int | None) -> np.random.RandomState:
    """Get rng"""
    return np.random.RandomState(random_seed)


def _is1d(x: np.ndarray) -> None:
    """Verify that the input vector in 1d and \
        raise ValuError if otherwise."""
    nd = 0
    for d in range(x.ndim):
        nd += (x.shape[d] != 1)

    if nd > 1:
        msg = f'Input array must be 1d. Received array with {nd} dimensions.\n' \
              'Please check your inputs.'
        logger.error(msg)
        raise ValueError(msg)


def true_signal(x: np.ndarray) -> np.ndarray:
    """
    Given a set of features values, compute the corresponding \
        signal's true values.

    Parameters
    ----------
    x: np.ndarray
        Input array of features.

    Returns
    -------
        np.ndarray
    """
    y = np.cos(np.pi * x)
    # deg = 6
    # roots = _rng(6).uniform(-1, 1, deg)
    # y = np.stack([x - r for r in roots], axis=-1).prod(axis=-1)
    return y


def get_training(
    n_samp: int,
    noise: float | None = None,
    random_seed: int | None = RANDOM_SEED,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given the number of samples generate training data.

    Notes
    -----
    x iid U[-1, 1]
    y = np.cos(pi * x)

    Parameters
    ----------
    n_samp: int
        Number of training samples.
    random_seed: int | None
        Random seed number for reproducibility. Pass None for fully \
        randomized results.

    Returns
    -------
    x_train: np.ndarray
        Training features.
    y_train: np.ndarray
        Training targets.
    """
    rng = _rng(random_seed)
    x = rng.uniform(-1, 1, n_samp)

    y = true_signal(x)
    if noise is not None:
        y += stats.norm(0, noise).rvs(n_samp)

    return x, y


def zero_mean(x: np.ndarray) -> np.ndarray:
    """
    Given the array of inputs compute the mean function \
        of the gaussian process.

    Notes
    -----
    m(x) = 0

    Parameters
    ----------
    x: np.ndarray
        Input array.

    Returns
    -------
        np.ndarray
    """
    return np.zeros_like(x)


def kernel_rbf(x: np.ndarray, xp: np.ndarray | None = None) -> np.ndarray:
    """
    Given two arrays of inputs compute the covariance function \
        of the gaussian process.

    Notes
    -----
    k(x, x') = exp(.5*(x -x')**2)

    Parameters
    ----------
    x: np.ndarray
        First input array.
    xp: np.ndarray
        Second input array.

    Returns
    -------
        np.ndarray
    """
    xp = [xp, x][xp is None]
    x, xp = x.ravel(), xp.ravel()
    delta = x[:, None] - xp[None, :]
    return np.exp(-1 * delta**2 / 2)


# CLASS
class SimpleGP():
    """
    Simple uni-variate Gaussian process with zero-mean function \
        and without hyperparameter optimization.

    Attributes
    ----------
    cov_fun: (np.ndarray) -> np.ndarray
        Callable for the covariance function
    """

    def __init__(
        self,
        cov_fun: Callable[[np.ndarray], np.ndarray] = kernel_rbf,
    ):
        """Initialize object."""

        # self.mean_fun = mean_fun
        self.cov_fun = cov_fun

        # Internal
        self.x_train: np.ndarray = None  # Training feature
        self.y_train: np.ndarray = None  # Training target
        self.inv_cov: np.ndarray = None  # Inverse covariance matrix
        self.noise: float = None

    # Utils
    def _is_fitted(self) -> None:
        """
        Test if the GP is fitted and raise a NotFittedError if otherwise.

        Raises
        ------
            NotFittedError
        """
        tests = [
            x is None for x in [
                self.x_train, self.y_train, self.cov_fun
            ]
        ]
        if np.array(tests).any():
            msg = 'Model not fitted. Run the fit sequence first.'
            logger.error(msg)
            raise NotFittedError(msg)

    def sample_prior(
        self,
        X: np.ndarray,
        n_samp: int = 10,
        random_seed: int | None = RANDOM_SEED,
    ) -> np.ndarray:
        """
        Given the vector of features, generate samples form the prior \
            distribution.

        Parameters
        ----------
        X: np.ndarray
            Input array of features.
        n_samp:int
            Number of samples to generate.
        random_seed: int | None
            Random seed number for reproducibility. Pass None for fully \
            randomized results.
        """
        _is1d(X)
        y = (
            stats
            .multivariate_normal(
                zero_mean(X),
                kernel_rbf(X),
                allow_singular=True,
            )
            .rvs(
                n_samp,
                random_state=_rng(random_seed)
            )
            .T
        )

        return y

    def compute_posterior(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Given the new (unseen) features, compute the posterior \
            distribution parameters.

        Parameters
        ----------
        X: np.ndarray
            Input array of features.

        Returns
        -------
        mean: np.ndarray
            Mean of the GP predictions.
        cov: np.ndarray
            Covariance matrix.
        """
        self._is_fitted()
        Ct = self.cov_fun(X, self.x_train)
        CtA = Ct @ self.inv_cov
        # mean = self.mean_fun(X) + CtA @ (self.y_train - self.mean_fun(self.x_train))
        mean = CtA @ self.y_train
        cov = self.cov_fun(X, X) - CtA @ Ct.T
        # cov += self.noise * np.eye(len(mean))

        return mean, cov

    def sample_posterior(
        self,
        X: np.ndarray,
        n_samp: int = 10,
        add_noise: bool = False,
        random_seed: int | None = RANDOM_SEED,
    ) -> np.ndarray:
        """
        Given the vector of features, generate samples form the prior \
            distribution.

        Parameters
        ----------
        X: np.ndarray
            Input array of features.
        n_samp:int
            Number of samples to generate.
        add_noise: bool
            An option for whether to add the measurement noise.
        random_seed: int | None
            Random seed number for reproducibility. Pass None for fully \
            randomized results.
        """
        _is1d(X)
        mean, cov = self.compute_posterior(X)
        cov += [1e-6, self.noise**2][add_noise] * np.eye(len(X))
        y = (
            stats
            .multivariate_normal(
                mean,
                cov,
                allow_singular=True,
            )
            .rvs(
                n_samp,
                random_state=_rng(random_seed)
            )
            .T
        )

        return y

    # Methods
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        noise: float | None = None
    ) -> 'SimpleGP':
        """
        Given the training data, fit the Gaussian Process.

        Parameters
        ----------
        X: np.ndarray
            Input array of features (must be 1d).
        y: np.ndarray
            Input array of targets (must be 1d).
        noise: float | None
            (Optional) Noise of the measurement data to account \
            for the fact that y_train values may not be 100% accurate.

        Returns
        -------
            SimpleGP
        """
        # Check
        _is1d(X), _is1d(y)

        # Assign
        X, y = X.ravel(), y.ravel()
        self.x_train, self.y_train = X, y

        cov = self.cov_fun(X, X)
        self.noise = [noise, 1e-6][noise is None]  # For numerical stability
        cov += self.noise**2 * np.eye(len(X))
        inv_cov = np.linalg.inv(cov)
        if not np.allclose(cov @ inv_cov, np.eye(cov.shape[0])):
            msg = 'Something went wrong with the matrix inversion.'
            logger.error(msg)
            raise ValueError(msg)
        self.inv_cov = inv_cov

        return self

    def predict(
        self,
        X: np.ndarray,
        return_std: bool = False,
        add_noise: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray | None]:
        """
        Given the array of (unseen) features, compute the GP \
            predictions.

        Parameters
        ----------
        X: np.ndarray
            Input array of features.
        return_std: bool
            An option for whether to return the standard deviations.
        add_noise: bool
            An option for whether to add the measurement noise.

        Returns
        -------
        pred: np.ndarray
            GP predictions.
        std: np.ndarray | None
            Standard deviations of the GP predictions.
        """
        _is1d(X)
        pred, cov = self.compute_posterior(X)
        var = cov.diagonal()
        if add_noise:
            var = var + self.noise**2
        # var += [0, self.noise**2][add_noise] * np.eye(len(pred))
        out = [pred, [pred, var**.5]][return_std]

        return out
