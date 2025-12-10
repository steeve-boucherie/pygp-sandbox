"""Generate dummy dataset for testing models."""
import logging
from typing import Tuple

import numpy as np

from scipy import stats


# LOGGER
logger = logging.getLogger(__name__)


# UTILS
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


# CLASS
class NoisyPCGenerator():
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
    noise: float
        Noise level (in fraction).
    seed: int
        Random seed number for reproducibility.
    """

    # Init
    def __init__(
        self,
        weib_a: float = 8,
        weib_k: float = 2.1,
        cp: float = 0.4,
        rd: float = 100,
        power_rated: float = 2500,
        rho: float = 1.2,
        noise: float = 0.1,
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
        self.noise = noise

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

    # Utils
    def _sample_ws(self, n_samp: int) -> np.ndarray:
        """
        Sample wind speed from the underlying distribution.

        Parameters
        ----------
        n_samp: int
            Number of samples.

        Returns
        -------
            np.ndarray
        """
        ws = self.ws_rvs.rvs(n_samp, random_state=self.rng)
        ws.sort()
        return ws

    def compute_power(self, ws: np.ndarray) -> np.ndarray:
        """
        Given the wind speed compute the corresponding turbine power.

        Parameters
        ----------
        ws: np.ndarray
            Array of wind speed values.

        Returns
        -------
            np.ndarray
        """

        power_aero = (.5 * self.cp * self.rho * self.swept_area * ws**3) / 1000

        return np.where(power_aero <= self.power_rated, power_aero, self.power_rated)

    def _sample_noise(self, ws: np.ndarray) -> np.ndarray:
        """
        Given the wind speed compute the corresponding noise.

        Parameters
        ----------
        ws: np.ndarray
            Array of wind speed values.

        Returns
        -------
            np.ndarray
        """
        ws = np.array(ws)
        eps = stats.norm.rvs(
            loc=0,
            scale=self.noise * ws,
            size=ws.shape,
            random_state=self.rng
        )
        return eps

    # Main
    def generate(self, n_samp: int = 5000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Given the number of samples, generate the noisy power curve data.

        Parameters
        ----------
        n_samp: int
            Number of samples.

        Returns
        -------
        ws: np.ndarray
            Array of noisy wind speed data.
        power: np.ndarray
            Array of power data.
        ws_true: np.ndarray
            Array of true wind speed data.
        """
        ws_true = self._sample_ws(n_samp)
        power = self.compute_power(ws_true)
        noise = self._sample_noise(ws_true)
        ws = ws_true + noise

        return ws, power, ws_true
