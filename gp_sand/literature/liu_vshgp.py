"""
Implemention of Spare Variational Heteroskedastic Gaussian Process \
    Liu et al. 2020.

Description
-----------
The model uses two GPs:
- f(x) ~ GP(0, K_f): latent function
- g(x) ~ GP(μ_0, K_g): log-noise process (noise variance = exp(g(x)))

The variational distribution q(g) = N(g|μ, Σ) is parameterized by
a diagonal matrix Λ through:
    μ = K_g(Λ - 0.5*I)1 + μ_0*1
    Σ^{-1} = K_g^{-1} + Λ

Reference
---------
arXiv:1811.01179v3

TODO: Finish description.
"""
import logging
from typing import Any, Mapping

from gpytorch.distributions import base_distributions, MultivariateNormal
from gpytorch.kernels import Kernel
from gpytorch.likelihoods import Likelihood
from gpytorch.means import Mean
from gpytorch.models import ApproximateGP
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    VariationalStrategy,
)

from linear_operator.operators import DiagLinearOperator

import numpy as np

import torch
from torch import Tensor


# PACKAGE IMPORTS
from gp_sand.utils import to_tensor


# LOGGER
logger = logging.getLogger(__name__)


# UTILS
# Nothin' :)


# HELPER CLASSES
class SparseGP(ApproximateGP):
    """Implementation of canonical sparse GP for tidier \
        implementation in the full code."""

    def __init__(
        self,
        inducing_points: np.ndarray | Tensor,
        mean_module: Mean,
        covar_module: Kernel,
    ):
        inducing_points = to_tensor(inducing_points)
        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=inducing_points.size(0)
        )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True  # optimises X_m jointly
        )
        super().__init__(variational_strategy)
        self.mean_module = mean_module
        self.covar_module = covar_module

    def forward(self, x: Tensor):
        latent_dist = MultivariateNormal(
            self.mean_module(x),
            self.covar_module(x)
        )
        return latent_dist


class HeteroskedasticLikelihood(Likelihood):
    """
    Likelihood for VSHGP (Liu et al. 2020).

    Holds a reference to the noise GP (g) and uses it for:
      - Training: computing R_g = diag(exp(mu_g - sigma2_g / 2)) for the ELBO
      - Prediction: adding exp(mu_g + sigma2_g / 2) to the predictive variance

    Parameters
    ----------
    noise_model : ApproximateGP
        The sparse GP for the log-noise function g.
    """

    def __init__(self, noise_model: ApproximateGP):
        """Init class"""
        super().__init__()
        self.noise_model = noise_model

    # Utils
    def added_noise(self, x: Tensor) -> DiagLinearOperator:
        """
        Given the feature tensor, compute the added noise to be added \
            to the posterior distribution from the latent GP (f).

        Parameters
        ----------
        x: Tensor
            The values of the training features to use for calling \
            the noise GP (g).

        Returns
        -------
            DiagLinearOperator
        """
        dist_g = self.noise_model(x)
        mu_g = dist_g.mean
        var_g = dist_g.variance  # Directly gives sigma**2
        
        return DiagLinearOperator(torch.exp(mu_g + .5 * var_g))

    # Methods
    def expected_log_prob(
        self,
        observations: Tensor,
        f_dist: MultivariateNormal,
        x: Tensor,
        *args: Any,
        **kwargs: Mapping[str, Any]
    ) -> Tensor:
        """
        Given the observations, and the posterior distribution of the \
            latent function, from GP (f), compute the log-likelihood term \
            to be used in the ELBO calculation.

        Notes
        -----
        >>> ll = log(N(y | 0, Qnn^f + R_g))

        Where:
        - `Qnn^f is the full posterior covariance of the sparse latent GP.
        - `R_g` is the noise precision term to be added on the diagonal \
            of the covariance matrix.

        >>> R_g = exp(mu_g - .5 * sigma_g**2)

        Attributes
        ----------
        observation: Tensor
            The values of the obervation of the training data.
        f_dist: MultivariateNormal
            Posterior distribution from the latent GP (f).
        x: Tensor
            The values of the training features to use for calling \
            the noise GP (g).

        Returns
        -------
            Tensor
        """
        # TODO: Handle nans ?

        # Call the noise GP
        dist_g = self.noise_model(x)
        mu_g = dist_g.mean
        var_g = dist_g.variance  # Directly gives sigma**2
        r_g = DiagLinearOperator(torch.exp(mu_g - .5 * var_g))

        # Add the diagonal
        dist = MultivariateNormal(
            f_dist.mean,
            f_dist.lazy_covariance_matrix + r_g
        )

        return dist.log_prob(observations)

    def forward(
        self,
        function_samples: Tensor,
        x: Tensor
    ) -> MultivariateNormal:
        """
        Given the posterior distribution of the latent GP (f) and the \
            feature values for unseen data, add the expected heteroskedastic noise.

        Parameters
        ----------
        function_samples: Tensor
            Samples of the posterior distribution for the unseen data \
            from the latent GP (f).
        x: Tensor
            The feature values on which the noise GP (g) should called.

        Returns
        -------
            MultivariateNormal
        """
        # Get the noise GP (g) predictions and the diag term
        noise = self.added_noise(x).diagonal(dim1=-1, dim2=-2)
        y_dist = base_distributions.Normal(
            function_samples,
            noise.sqrt()
        )

        return y_dist

    def marginal(
        self,
        f_dist: MultivariateNormal,
        x: Tensor
    ) -> MultivariateNormal:
        """
        Given the posterior distribution of the latent GP (f) and the \
            feature values for unseen data, compute the posterior marginal distribution.
        Parameters
        ----------
        function_samples: Tensor
            Samples of the posterior distribution for the unseen data \
            from the latent GP (f).
        x: Tensor
            The feature values on which the noise GP (g) should called.

        Returns
        -------
            MultivariateNormal
        """
        mean, covar = f_dist.mean, f_dist.lazy_covariance_matrix
        noise_covar = self.added_noise(x)
        full_covar = covar + noise_covar
        return MultivariateNormal(mean, full_covar)
