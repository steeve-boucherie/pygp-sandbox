"""Generate dummy dataset for testing models."""
import logging

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
