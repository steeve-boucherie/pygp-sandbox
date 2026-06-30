"""Custom Distribution classes"""
import logging

import torch
from torch import Tensor
from torch.distributions import Gumbel


# LOGGER
logger = logging.getLogger(__name__)


# DISTRIBUTIONS
class GumbelDistribution(Gumbel):
    """
    Extension of Torch's Gumbel distribution to include \
    the confidence region method and safe log_prob method.

    Attributes
    ----------
    loc: Tensor | float
        Value(s) for the location parameter of the distribution, mu.
    scale: Tensor | float
        Value(s) for the scale parameter of the distribution, beta.
    z_clamp: float | None
        (Optional) Clipping value to avoid overflow in estimation \
        of the log-probability.
    """

    def __init__(
        self,
        loc: Tensor | float,
        scale: Tensor | float,
        z_clamp: float | None = 15.,
        validate_args: bool | None = None,
    ) -> None:
        super().__init__(loc, scale, validate_args)
        self.z_clamp = z_clamp

    # Method
    def confidence_region(
        self,
        p_lower: float = 2.5,
        p_upper: float = 97.5,
    ) -> tuple[Tensor, Tensor]:
        """
        Returns the lower and upper values defining the confidence region.

        Notes
        -----
        Percentile thresholds must be provided in PERCENT.

        Parameters
        ----------
        p_lower: float
            Percentile values defining the lower bound of the interval. \
            Default is 2.5%.
        p_upper: float
            Percentile values defining the upper bound of the interval. \
            Default is 97.5%.

        Returns
        -------
        lower: Tensor
            Tensor of lower bounds values.
        upper: Tensor
            Tensor of upper bounds values.
        """
        p_lower = torch.tensor(p_lower / 100)
        p_upper = torch.tensor(p_upper / 100)
        lower = self.loc - self.scale * torch.log(-torch.log(p_lower))
        upper = self.loc - self.scale * torch.log(-torch.log(p_upper))
        return lower, upper

    def log_prob(self, value: Tensor):
        if self._validate_args:
            self._validate_sample(value)
        z = (self.loc - value) / self.scale
        if self.z_clamp:
            z = z.clip(-self.z_clamp, self.z_clamp)
        return (z - z.exp()) - self.scale.log()
