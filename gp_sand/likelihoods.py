"""Customized mean modules for GPs regression."""
import logging
from typing import Any

from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import _GaussianLikelihoodBase, GaussianLikelihood
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


# FIXME: Remove since this is not needed.
class VariationalHeteroscedasticLikelihood(GaussianLikelihood):
    """
    Custom likelihood for VHGP. This is mostly a placeholder since \
        we'll implement the actual objective in the MLL class.
    """
    def __init__(self):
        super().__init__()

    def forward(self, function_samples, *args, **kwargs):
        # This won't be used in practice
        return function_samples

    def marginal(
        self,
        function_dist: MultivariateNormal,
        *params: Any,
        **kwargs: Any
    ) -> MultivariateNormal:
        # mean, covar = function_dist.mean, function_dist.lazy_covariance_matrix
        # noise_covar = self._shaped_noise_covar(mean.shape, *params, **kwargs)
        # full_covar = covar + noise_covar
        return function_dist
