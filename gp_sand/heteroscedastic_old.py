"""Class and method for handling GP with heteroscedastic noise models"""
import logging


import gpytorch
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from gpytorch.models import ExactGP
from gpytorch.likelihoods import _GaussianLikelihoodBase, GaussianLikelihood
from gpytorch.likelihoods.noise_models import HeteroskedasticNoise

from torch import Tensor

# LOGGER
logger = logging.getLogger(__name__)


# LATENT
class NoiseGP(ExactGP):
    """
    Latent Gaussian Process to fit to the noise data.

    Attributes
    ----------
    train_x: Tensor
        Tensor of training input features.
    train_y: Tensor
        Tensor of training input targets.
    likelihood: GaussianLikelihood
        Likelihood model for the observation noise.
    """

    def __init__(
        self,
        train_x: Tensor = None,
        train_y: Tensor = None,
        likelihood: GaussianLikelihood = None,
    ):
        """Create class instance."""
        super(NoiseGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        # self.covar_module = ScaleKernel(RBFKernel())
        self.covar_module = ScaleKernel(MaternKernel())

    def forward(self, x: Tensor) -> MultivariateNormal:
        """Forward input."""
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


# LIKELIHOOD
class HeteroskedasticGaussianLikelihood(_GaussianLikelihoodBase):
    """Wrapper class around the base Gaussian likelihood model."""
    def __init__(self, noise_covar: HeteroskedasticNoise, **kwargs) -> None:
        super().__init__(noise_covar, **kwargs)


# OVER-ARCHING GP
class HeteroskedasticGP(ExactGP):
    """
    Standard Exact GP with heteroskedastic noise.

    Attributes
    ----------
    train_x: Tensor
        Tensor of training input features.
    train_y: Tensor
        Tensor of training input targets.
    likelihood: HeteroskedasticGaussianLikelihood
        Likelihood model for the heteroskedastic observation noise.
    """

    def __init__(
        self,
        train_x: Tensor = None,
        train_y: Tensor = None,
        likelihood: HeteroskedasticGaussianLikelihood = None,
    ):
        """Create class instance."""
        super(HeteroskedasticGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel())
        # self.covar_module = ScaleKernel(MaternKernel())

    def forward(self, x: Tensor) -> MultivariateNormal:
        """Forward input."""
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
