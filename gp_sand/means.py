"""Customized mean modules for GPs regression."""
import logging

from gpytorch.means import Mean

import torch
from torch import nn, Tensor


# PACKAGE IMPORT
# Nothin'


# LOGGER
logger = logging.getLogger(__name__)


# CUSTOM MEAN MODULE
class HyperbolicTangentMean(Mean):
    """
    Paramettric mean function to use in GP,
        approximating it as an hyperbolixc tangent.

    The power curve is modelled as followed
    >>> p(x) = 0.5 * scale * (np.tanh(shape * (x - loc)) + 1)

    Where: scale, shape and loc are learnable parameters.

    Attributes
    ----------
    scale: Parameter | None
        Initial value for the scale parameter.
    shape: Parameter | None
        Initial value for the shape parameter.
    loc: Parameter | None
        Initial value for the loc parameter.
    """
    _LOC = nn.Parameter(torch.tensor(0.))
    _SCALE = nn.Parameter(torch.tensor(1.))
    _SHAPE = nn.Parameter(torch.tensor(1.))

    def __init__(
        self,
        scale: nn.Parameter | None = None,
        shape: nn.Parameter | None = None,
        loc: nn.Parameter | None = None,

    ):
        super().__init__()
        self.loc = [loc, self._LOC][loc is None]
        self.scale = [scale, self._SCALE][scale is None]
        self.shape = [shape, self._SHAPE][shape is None]

    def forward(self, x: Tensor):
        """
        Given the wind speed, compute the power.

        Parameters
        ----------
        x: tensor, shape (n,)
            wind speed values, shape (n,) or (n, 1)

        Returns
        -------
            tensor, shape (n,)
        """
        p_norm = 0.5 * (torch.tanh(self.shape * (x - self.loc)) + 1)
        p_norm = p_norm.clamp(min=1e-3, max=1.0)
        power = self.scale * p_norm
        return power.squeeze()
