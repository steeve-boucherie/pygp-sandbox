"""Generate dummy dataset for testing models."""
import abc
import logging
from typing import Tuple

import numpy as np

from scipy import stats

import torch
from torch import Tensor


# LOGGER
logger = logging.getLogger(__name__)


# UTILS
def _rng(random_seed: int | None) -> np.random.RandomState:
    """Get rng"""
    return np.random.RandomState(random_seed)


def homoscedastic_data(
    n_points: int = 1000,
    t_df: int = 3,
    t_loc: float = 0.0,
    t_scale: float = 2.1,
    m: float = 3.,
    sigma: float = 0.1,
    seed: int = 42
) -> np.ndarray:
    """
    Generate synthetic heteroskedastic data.

    Notes
    -----
    - Sample x from a Student's t-distribution.
    - Use power low of cosine(x) for y true-values.
    - Apply white noise - normally distributed (iid)

    Parameters:
    -----------
    n_points : int
        Number of data points
    t_df : float
        Degrees of freedom for Student's t distribution
    t_loc : float
        Location parameter (mean) for Student's t
    t_scale : float
        Scale parameter for Student's t
    m: float
        The power exponent to apply to the cosine of x.
    sigma: float
        The standard deviation of the white noise.
    seed : int
        Random seed for reproducibility
    """
    rng = np.random.RandomState(seed)

    # Generate x from Student's t distribution
    x = stats.t.rvs(
        t_df,
        loc=t_loc,
        scale=t_scale,
        size=n_points,
        random_state=rng
    )
    # x = x[(-15 <= x) & (x <= 15)]
    x = np.sort(x)  # Sort for nicer visualization

    # True function: power cosine

    y_true = np.power(np.cos(np.deg2rad(x)), m)

    # # Heteroskedastic noise: sigma depends on x
    # # Use log(|x| + 1) to avoid issues with x near 0
    # sigma_x = 0.1 + 0.3 * np.log(np.abs(x) + 1)
    # noise = np.random.normal(0, sigma_x)

    # Homoscedastic noise
    noise = stats.norm.rvs(0, sigma, size=n_points, random_state=rng)

    # Add heteroskedastic noise
    y = y_true + noise

    return x, y, y_true  # , sigma_x


def heteroscedastic_data(
    n_points: int = 1000,
    t_df: int = 3,
    t_loc: float = 0.0,
    t_scale: float = 2.1,
    m: float = 3.,
    sigma: float = 0.1,
    seed: int = 42
) -> np.ndarray:
    """
    Generate synthetic heteroskedastic data.

    Notes
    -----
    - Sample x from a Student's t-distribution.
    - Use power low of cosine(x) for y true-values.
    - Apply white noise - normally distributed (iid)

    Parameters:
    -----------
    n_points : int
        Number of data points
    t_df : float
        Degrees of freedom for Student's t distribution
    t_loc : float
        Location parameter (mean) for Student's t
    t_scale : float
        Scale parameter for Student's t
    m: float
        The power exponent to apply to the cosine of x.
    sigma: float
        The standard deviation of the white noise.
    seed : int
        Random seed for reproducibility
    """
    rng = np.random.RandomState(seed)

    # Generate x from Student's t distribution
    x = rng.uniform(-15, 15, n_points)

    # True function: power cosine
    y_true = np.power(np.cos(np.deg2rad(x)), m)

    # Heteroskedastic noise: sigma depends on x
    # sigma = 0.02 * (1 + np.log10(np.abs(x) + 1))
    sigma = 0.001 * (1 + np.abs(x))

    # Homoscedastic noise
    noise = stats.norm.rvs(0, sigma, size=n_points, random_state=rng)

    # Add heteroskedastic noise
    y = y_true + noise

    return x, y, y_true  # , sigma_x


# ABSTRACT
class SyntheticDataGeneratorInterface(abc.ABC):
    """
    Abstract class for implementing synthetic data generator \
        with support for heteroscedastic noise.

    Abstract Methods
    ----------------
    All inherting classes must implement the following method.
    true_signal: [np.ndarray] -> np.ndarray
        Given the feature(s), compute the signal's true values.
    noise: [np.ndarray] -> np.ndarray
        Given the feature(s), compute the observation noise \
        expressed as standard deviation (Gaussian).
    sample_x: [int] -> np.ndarray
        Given the expect number of samples, sample the values of \
        the feature.
    training_data: [int] -> Tuple[np.ndarray, np.ndarray]
        Given the expect number of samples, generate a pair of \
        feature and observations for model training.
    """

    # Abstract method
    @abc.abstractmethod
    def true_signal(self, X: np.ndarray) -> np.ndarray:
        """Given the feature(s), compute the signal's true values."""
        raise NotImplementedError('This in abstract method.')

    @abc.abstractmethod
    def noise(self, X: np.ndarray) -> np.ndarray:
        """Given the feature(s), compute the observation noise magnitude \
            expressed as standard deviation (Gaussian)."""
        raise NotImplementedError('This in abstract method.')

    @abc.abstractmethod
    def generate_noise(self, X: np.ndarray) -> np.ndarray:
        """Given the feature data, generate the corresponding observation \
            noise samples."""

    @abc.abstractmethod
    def sample_x(self, n_samples: int) -> np.ndarray:
        """Given the expect number of samples, sample the values of \
            the feature."""
        raise NotImplementedError('This in abstract method.')

    @abc.abstractmethod
    def training_data(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Given the expect number of samples, generate a pair of \
            feature and observations for model training."""
        raise NotImplementedError('This in abstract method.')


# CLASS
class NoisyPCGenerator(SyntheticDataGeneratorInterface):
    """
    Class to generate noisy power curve data.

    Attributes
    ----------
    weib_a: float
        Scale factor of the weibull distribution for the wind speed.
    weib_k: float
        Shape factor of the weibull distribution for the wind speed.
    cp: float
        Wind turbine's (rated) power coefficient.
    power_rated: float
        Turbine's rated power in kW.
    rd: float
        Rotor diameter.
    rho: float
        Air density value. Default = 1.2 kg.m^-3
    cov: float
        Noise level (in fraction).
    seed: int
        Random seed number for reproducibility.
    """

    # Default number of samples
    _NSAMP = 201

    # Init
    def __init__(
        self,
        weib_a: float = 8,
        weib_k: float = 2.1,
        cp: float = 0.4,
        rd: float = 100,
        power_rated: float = 2500,
        rho: float = 1.2,
        cov: float = 0.1,
        seed: int = 1977,
    ):
        """Instantiate class object."""
        # Atmospheric cond
        self.weib_a = weib_a
        self.weib_k = weib_k
        self.rho = rho

        # Turbine detail
        self.cp = cp
        self.rd = rd
        self.power_rated = power_rated

        # Noise info
        self.cov = cov

        self.rng = np.random.RandomState(seed)

    # Properties
    @property
    def swept_area(self) -> float:
        """Return the rotor swept area."""
        return np.pi * self.rd**2 / 4

    @property
    def ws_rvs(self) -> stats.rv_continuous:
        """Return the RV wind speed distribution"""
        return stats.weibull_min(self.weib_k, scale=self.weib_a)

    # Methods
    def true_signal(self, X: np.ndarray) -> np.ndarray:
        """
        Given the feature(s), compute the signal's true values.

        Parameters
        ----------
        X: np.ndarray
            Input features values (wind speed).

        Returns
        -------
            np.ndarray
        """
        power_aero = .5 * self.cp * self.rho * self.swept_area * X**3
        power_aero /= 1000
        power_aero = np.where(
            power_aero <= self.power_rated,
            power_aero,
            self.power_rated
        )
        return power_aero

    def noise(self, X: np.ndarray) -> np.ndarray:
        """
        Given the feature(s), compute the observation noise \
            expressed as standard deviation (Gaussian).

        Notes
        -----
        >>> std = cov * ws

        Where cov is the noise level specified as class attribute.

        Parameters
        ----------
        X: np.ndarray
            Input features values (wind speed).

        Returns
        -------
            np.ndarray
        """
        return self.cov * np.array(X)

    def generate_noise(
        self,
        X: np.ndarray,
        random_seed: int | None = None,
    ) -> np.ndarray:
        """
        Given the feature data, generate the corresponding observation \
            noise values.

        Parameters
        ----------
        X: np.ndarray
            Input features values (wind speed).
        random_seed: int | None
            (Optional) Random seed number for reproducibility \
            use the class default if None is passed.

        Returns
        -------
            np.ndarray
        """
        rng = [self.rng, _rng(random_seed)][random_seed is not None]
        eps = stats.norm.rvs(
            loc=0,
            scale=self.noise(X),
            size=X.shape,
            random_state=rng
        )
        return eps

    def sample_x(
        self,
        n_samples: int | None = None,
        random_seed: int | None = None,
    ) -> np.ndarray:
        """
        Given the expect number of samples, sample the values of \
            the feature.

        Parameters
        ----------
        n_samples: np.ndarray
            Number of samples to be generated.
        random_seed: int | None
            (Optional) Random seed number for reproducibility \
            use the class default if None is passed.

        Returns
        -------
            np.ndarray
        """
        n_samples = [self._NSAMP, n_samples][n_samples is not None]
        rng = [self.rng, _rng(random_seed)][random_seed is not None]

        x = (
            stats
            .weibull_min(
                self.weib_k,
                scale=self.weib_a
            )
            .rvs(n_samples, random_state=rng)
        )
        return x

    def training_data(
        self,
        n_samples: int | None = None,
        to_torch: bool = True,
        random_seed: int | None = None,
    ) -> Tuple[np.ndarray | Tensor, np.ndarray | Tensor]:
        """
        Given the expect number of samples, generate a pair of \
            features and observations for model training.

        Parameters
        ----------
        n_samples: np.ndarray
            Number of samples to be generated.
        to_torch: bool
            An option for whether to output the training data in torch \
            format (tensor) directly. Use torch.float32 is set to True.
        random_seed: int | None
            (Optional) Random seed number for reproducibility \
            use the class default if None is passed.

        Returns
        -------
        x_train: np.ndarray | Tensor
            Training features.
        y_train: np.ndarray | Tensor
            Training observations.
        """
        x_train = self.sample_x(n_samples, random_seed)
        y_train = self.true_signal(x_train)
        x_train += self.generate_noise(x_train)

        if to_torch:
            x_train = torch.tensor(x_train).to(torch.float32)
            y_train = torch.tensor(y_train).to(torch.float32)

        return x_train, y_train


class PowerCosineHeteroscedastic(SyntheticDataGeneratorInterface):
    """
    Synthetic data generator for power cosine with heteroscedastic \
        (linear) noise.

    Noise
    -----
    Noise is computed as a linear function of the absolute value of \
        the angle in DEGREES.

    Attributes
    ----------
    cos_m: float
        Value of the power exponent to apply to the cosinus values.
    noise_slope: float
        Value of the slope for noise level.
    noise_intercept: float
        Value of the intercept for noise level.
    random_seed: RandomState
    """

    # Default number of samples
    _NSAMP = 201

    # Init
    def __init__(
        self,
        cos_m: float = 3.,
        noise_slope: float = 0.1,
        noise_intercept: float = 0.05,
        random_seed: int | None = 1977,
    ):
        """Init class object."""
        self.cos_m = cos_m
        self.noise_slope = noise_slope
        self.noise_intercept = noise_intercept

        # Randomness
        self.rng = _rng(random_seed)

    # Method
    def true_signal(self, X: np.ndarray) -> np.ndarray:
        """
        Given the feature(s), compute the signal's true values.

        Notes
        -----
        Expects inputs in DEGREES.
        Expects normalized inputs. The feature values are scaled by a factor \
            of 2pi and input to the cosine expression.

        Parameters
        ----------
        X: np.ndarray
            Input features values in DEGREES.

        Returns
        -------
            np.ndarray
        """
        return np.pow(np.cos(np.deg2rad(X)), self.cos_m)

    def noise(self, X: np.ndarray) -> np.ndarray:
        """
        Given the feature(s), compute the observation noise \
            expressed as standard deviation (Gaussian).

        Notes
        -----
        Expects inputs in DEGREES.

        >>> noise = (a * |x| + b)**2

        Parameters
        ----------
        X: np.ndarray
            Input features values in DEGREES.

        Returns
        -------
            np.ndarray
        """
        xn = X / np.abs(X).max()
        return np.pow(self.noise_intercept + self.noise_slope * np.abs(xn)**3, 2)

    def generate_noise(
        self,
        X: np.ndarray,
        random_seed: int | None = None,
    ) -> np.ndarray:
        """
        Given the feature data, generate the corresponding observation \
            noise values.

        Parameters
        ----------
        X: np.ndarray
            Input features values in DEGREES.
        random_seed: int | None
            (Optional) Random seed number for reproducibility \
            use the class default if None is passed.

        Returns
        -------
            np.ndarray
        """
        rng = [self.rng, _rng(random_seed)][random_seed is not None]
        sigma = self.noise(X)**.5
        return stats.norm(0, sigma).rvs(X.shape[0], random_state=rng)

    def sample_x(
        self,
        n_samples: int | None = None,
        random_seed: int | None = None,
    ) -> np.ndarray:
        """
        Given the expect number of samples, sample the values of \
            the feature.

        Parameters
        ----------
        n_samples: np.ndarray
            Number of samples to be generated.
        random_seed: int | None
            (Optional) Random seed number for reproducibility \
            use the class default if None is passed.

        Returns
        -------
            np.ndarray
        """
        n_samples = [self._NSAMP, n_samples][n_samples is not None]
        rng = [self.rng, _rng(random_seed)][random_seed is not None]
        return rng.uniform(-15, 15, n_samples)

    def training_data(
        self,
        n_samples: int | None = None,
        to_torch: bool = True,
        random_seed: int | None = None,
    ) -> Tuple[np.ndarray | Tensor, np.ndarray | Tensor]:
        """
        Given the expect number of samples, generate a pair of \
            features and observations for model training.

        Parameters
        ----------
        n_samples: np.ndarray
            Number of samples to be generated.
        to_torch: bool
            An option for whether to output the training data in torch \
            format (tensor) directly. Use torch.float32 is set to True.
        random_seed: int | None
            (Optional) Random seed number for reproducibility \
            use the class default if None is passed.

        Returns
        -------
        x_train: np.ndarray | Tensor
            Training features.
        y_train: np.ndarray | Tensor
            Training observations.
        """
        x_train = self.sample_x(n_samples, random_seed)
        y_train = self.true_signal(x_train)
        y_train += self.generate_noise(x_train)

        if to_torch:
            x_train = torch.tensor(x_train).to(torch.float32)
            y_train = torch.tensor(y_train).to(torch.float32)

        return x_train, y_train
