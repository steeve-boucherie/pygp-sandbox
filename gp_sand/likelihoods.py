"""Customized mean modules for GPs regression."""
import logging
from typing import Any, Mapping

from gpytorch.distributions import MultivariateNormal
from gpytorch.likelihoods import (
    _GaussianLikelihoodBase,
    _OneDimensionalLikelihood,
    GaussianLikelihood,
)
from gpytorch.likelihoods.noise_models import HeteroskedasticNoise

import torch
from torch import Tensor
# from torch.distributions import Normal


# PACKAGE IMPORTS
from gp_sand.distributions import GumbelDistribution


# LOGGER
logger = logging.getLogger(__name__)


# HOMOSCEDASTIC NOISE
class GumbelLikelihood(_OneDimensionalLikelihood):
    def __init__(self):
        super().__init__()
        # Learnable log-scale (ensures scale > 0)
        self.raw_scale = torch.nn.Parameter(torch.zeros(1))

    @property
    def scale(self) -> Tensor:
        return torch.nn.functional.softplus(self.raw_scale)

    def forward(
        self,
        f: MultivariateNormal,
        **kwargs: Mapping[str, Any]
    ) -> GumbelDistribution:
        return GumbelDistribution(loc=f, scale=self.scale)

    def marginal(
        self,
        function_dist: MultivariateNormal,
        *params: Any,
        **kwargs: Mapping[str, Any]
    ) -> GumbelDistribution:
        return GumbelDistribution(function_dist.mean, self.scale)

    def expected_log_prob(
        self,
        y: Tensor,
        f: MultivariateNormal,
        **kwargs: Mapping[str, Any]
    ) -> Tensor:
        return super().expected_log_prob(y, f, **kwargs)


# CUSTOM LIKELIHOOD
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
