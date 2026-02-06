"""Customized mean modules for GPs regression."""
import logging
# from typing import Any

# from gpytorch.distributions import base_distributions
from gpytorch.likelihoods import _GaussianLikelihoodBase
from gpytorch.likelihoods.noise_models import HeteroskedasticNoise

# from torch import Tensor
# from torch.distributions import Normal


# PACKAGE IMPORTS
# Nothin'


# LOGGER
logger = logging.getLogger(__name__)


# CUSTOME LIKELIHOOD
class HeteroskedasticGaussianLikelihood(_GaussianLikelihoodBase):
    """
    Gaussian Likelihood model handling heteroskedastic noise.

    Notes
    -----
    Wrapper around GPyTorch class. Uses underlying GP to make predictions \
        on the noise level.

    Attributes
    ----------
    noise_covar: HeteroskedasticNoise
        Instance of hetereoskedastic noise model.
    """
    def __init__(self, noise_covar: HeteroskedasticNoise, **kwargs) -> None:
        super().__init__(noise_covar, **kwargs)
