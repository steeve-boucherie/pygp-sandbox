"""
Implemention of Stochastic Variational Sparse Heteroskedastic Gaussian Process \
    Liu et al. 2020.

Description
-----------
The model uses two GPs:
- f(x) ~ GP(0, K_f): latent function
- g(x) ~ GP(μ_0, K_g): log-noise process (noise variance = exp(g(x)))

Two variational distributions are used independtly:
- p(f) = N(f | mu_f, Sigma_f) for approximating the latent function GP.
- q(g) = N(q | mu_g, Sigma_g) for approximating the noise GP.

This formulation allows for mini-batching to speed training on large datasets.

Reference
---------
arXiv:1811.01179v3

TODO: Finish description.
"""
import logging
from typing import Any, Callable, List, Mapping, Tuple

from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import Kernel, RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean, Mean, ZeroMean
from gpytorch.mlls import AddedLossTerm
from gpytorch.mlls._approximate_mll import _ApproximateMarginalLogLikelihood
from gpytorch.models import ApproximateGP
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    VariationalStrategy
)

from linear_operator.operators import DiagLinearOperator

import numpy as np

import pandas as pd

import torch
from torch import Tensor
from torch.distributions import Normal
from torch.optim import Adam


# PACKAGE IMPORTS
# from gp_sand.means import MeanInterface
from gp_sand.metrics import compute_scores, display_scores
from gp_sand.models import GPInterface, SCORES
from gp_sand.utils import (to_numpy, to_tensor)


# LOGGER
logger = logging.getLogger(__name__)


# UTILS
def default_kernel(d: int) -> Kernel:
    """
    Get the default kernel used in the paper.

    Parameters
    ----------
    d: int
        Number of features.

    Returns
    -------
        Kernel
    """
    return ScaleKernel(RBFKernel(ard_num_dims=d))


# ADDED LOSS TERMS
class TraceGAddedLoss(AddedLossTerm):
    """
    Added loss term for the trace of the noise GP variance prediction in \
        the SVSHGP formulation (Liu et al. 2020, Eq. 18).

    >>> loss = 0.25 * trace(Sigma_g)

    Where:
    - Sigma_g is the covariance matrix of the noise-GP estimated on the \
        the training data.

    Attributes
    ----------
    sigma_g: Tensor
        Posterior predictive for the variance of the latent GP.
    """

    def __init__(self, sigma_g: Tensor):
        self.sigma_g = sigma_g

    def loss(self, *args, **kwargs) -> Tensor:
        """Compute the added loss term."""
        loss = 0.25 * self.sigma_g
        return loss.sum() / loss.size(0)


class TraceFAddedLoss(AddedLossTerm):
    """
    Added loss term for the trace of the latent GP variance prediction in \
        the SVSHGP formulation (Liu et al. 2020, Eq. 18).

    >>> loss = 0.5 * (trace(Sigma_g) @ R_g^-1)

    With:
    >>> R_g = exp(mu_g - 0.5 * Sigma_g)

    Where:
    - Sigma_f is the covariance matrix of the latent-GP estimated on the \
        the training data.
    - mu_g is the mean vector matrix of the noise-GP estimated on the \
        the training data.
    - Sigma_g is the covariance matrix of the noise-GP estimated on the \
        the training data.

    Attributes
    ----------
    sigma_f: Tensor
        Posterior predictive for the variance of the latent GP.
    rg_diag: Tensor
        Extra diagonal term from the noise-GP predictions.
    """

    def __init__(
        self,
        sigma_f: Tensor,
        rg: Tensor,
    ):
        self.sigma_f = sigma_f
        self.rg = rg

    def loss(self, *args, **kwargs) -> Tensor:
        """Compute the added loss term."""
        loss = 0.5 * (self.sigma_f / self.rg)
        return loss.sum() / loss.size(0)


# NOISE GP
class NoiseGP(ApproximateGP):
    """
    Variational Sparse Gaussian Process for the observation Noise.
    """

    def __init__(
        self,
        ind_points: Tensor,
        mean_module: Mean,
        covar_module: Kernel,
        jitter_val: float | None = None,
    ):
        """Init class."""
        var_dist = CholeskyVariationalDistribution(
            num_inducing_points=ind_points.size(0),
        )
        var_strat = VariationalStrategy(
            self,
            inducing_points=ind_points,
            variational_distribution=var_dist,
            learn_inducing_locations=True,
            jitter_val=jitter_val
        )
        super().__init__(var_strat)

        self.mean_module = mean_module
        self.covar_module = covar_module

    # Forward
    def forward(self, x: Tensor) -> MultivariateNormal:
        """
        Given the input  tesnor compute the multivariate normal distribution.

        Parameters
        ----------
        x: Tensor
            Input tensor.

        Returns
        -------
            MultivariateNormal
        """
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return MultivariateNormal(mean, covar)

    # Utils
    def rg_diag(self, x: Tensor) -> Tensor:
        """
        Given the feature tensor, compute the added diagonal term, Rg, \
            to used for model training during training.

        Parameters
        ----------
        x: Tensor
            The values of the training features to use for calling \
            the noise GP (g).

        Returns
        -------
            Tensor
        """
        g_dist = self(x)
        rg = torch.exp(g_dist.mean - .5 * g_dist.variance)

        return rg

    def added_noise(self, x: Tensor) -> DiagLinearOperator:
        """
        Given the feature tensor, compute the added noise to be added \
            to the posterior distribution from the latent GP (f).

        Parameters
        ----------
        x: Tensor
            The values of the unseen features to use for calling \
            the noise GP (g).

        Returns
        -------
            DiagLinearOperator
        """
        dist_g = self(x)
        mu_g = dist_g.mean
        var_g = dist_g.variance  # Directly gives sigma**2

        return DiagLinearOperator(torch.exp(mu_g + .5 * var_g))


# MLL
class SVSHGPVariationalELBO(_ApproximateMarginalLogLikelihood):
    """
    Variational ELBO marginal likelihood class for the Stochastic Variational \
        Sparse Heteroskedastic GP.

    Attributes
    ----------
    num_data: int
        The total number of training data points (used for normalization).
    beta: float
        A multiplicative factor for the KL divergence terms. Default is 1.
    combine_terms: bool
        Whether or not to sum the expected MLL with the KL terms. Default is True.
    """

    def __init__(
        self,
        model: 'SVSHGP',
        num_data: int,
        beta: float = 1,
        combine_terms: bool = True
    ):
        # NOTE: Use a dummy gaussian likelihood for super().__init__
        # This is ignored as the LL term is computed manually.
        super().__init__(GaussianLikelihood(), model, num_data, beta, combine_terms)

    def _log_likelihood_term(
        self,
        approximate_dist_f: MultivariateNormal,
        target: Tensor,
        **kwargs
    ):
        """
        Compute the log likelihood term of the ELBO.

        Attributes
        ----------
        approximate_dist_f: MultiVariateNormal
            Estimated MVN.
        target: Tensor
            Training target values.
        kwargs: Mapping[str, Any]
            Optional argument it must constain the tensor of training features \
            to estimate the diagonal term.
        """
        # Get the noise-GP predictions
        g_dist: MultivariateNormal = self.model.noise_gp(kwargs['x'])
        rg = torch.exp(g_dist.mean - 0.5 * g_dist.variance)

        # Term : main likelihood (residuals)
        llk = Normal(approximate_dist_f.mean, rg.sqrt()).log_prob(target).sum()

        # Term 2: trace of G
        trace_g = 0.25 * g_dist.variance.sum()

        # Term 3: trace of F scaled with the Rg posterior
        trace_f = 0.5 * (approximate_dist_f.variance / rg).sum()

        return llk - trace_g - trace_f


# STOCHASTIC VARIATIONAL SPARSE HETEROSKEDASTIC GAUSSIAN PROCESS
class SVSHGP(ApproximateGP, GPInterface):
    """
    Implemention of Stochastic Variational Sparse Heteroskedastic Gaussian Process \
        Liu et al. 2020.

    Description
    -----------
    TODO: Update - add mention that GP are sparse.
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


    Attributes
    ----------
    TODO: List class attributes
    """

    def __init__(
        self,
        ind_points_f: np.ndarray | Tensor,
        ind_points_g: np.ndarray | Tensor,
        mean_f: Mean = ZeroMean(),
        covar_f: Kernel | None = None,
        mean_g: Mean = ConstantMean(),
        covar_g: Kernel | None = None,
        jitter_val: float | None = None,
    ):
        """Init class."""
        # Set up the latent f GP
        ind_points_f = to_tensor(ind_points_f)
        var_dist = CholeskyVariationalDistribution(
            num_inducing_points=ind_points_f.size(0)
        )
        var_strat = VariationalStrategy(
            self,
            inducing_points=ind_points_f,
            variational_distribution=var_dist,
            learn_inducing_locations=True,
            jitter_val=jitter_val
        )

        super().__init__(var_strat)

        self.mean_module = mean_f
        self.covar_module = [
            covar_f,
            default_kernel(ind_points_f.size(1))
        ][covar_f is None]

        # Set up the noise GP
        ind_points_g = to_tensor(ind_points_g)
        self.noise_gp = NoiseGP(
            ind_points_g,
            mean_module=mean_g,
            covar_module=[
                covar_g,
                default_kernel(ind_points_g.size(1))
            ][covar_g is None],
            jitter_val=jitter_val
        )

        # Register added loss terms
        # self.register_added_loss_term('trace_g')
        # self.register_added_loss_term('trace_f')

    # Utils
    # def update_loss_terms(self, x: Tensor):
    #     """Update the added loss terms."""
    #     # sigma_g = self.noise_gp.covar_module(x)
    #     self.update_added_loss_term(
    #         'trace_g',
    #         TraceGAddedLoss(
    #             self.noise_gp(x).variance
    #         )
    #     )
    #     self.update_added_loss_term(
    #         'trace_f',
    #         TraceFAddedLoss(
    #             self(x).variance,
    #             self.noise_gp.rg_diag(x),
    #         )
    #     )

    # Forward
    def forward(self, x: Tensor) -> MultivariateNormal:
        """
        Given the input  tesnor compute the multivariate normal distribution.

        Parameters
        ----------
        x: Tensor
            Input tensor.

        Returns
        -------
            MultivariateNormal
        """
        mean = self.mean_module(x)
        covar = self.covar_module(x)

        return MultivariateNormal(mean, covar)

    # Fit/Predict
    def fit(
        self,
        train_x: np.ndarray | Tensor,
        train_y: np.ndarray | Tensor,
        n_epochs: int = 250,
        optim_kw: Mapping[str, Any] = {},
        verbose: bool = True,
    ) -> 'NoiseGP':
        """
        Given the traning data and fitting option, fit the model.

        Parameters
        ----------
        train_x: np.ndarray | torch.Tensor, shape (n, m)
            Tensor of training features.
        train_y: np.ndarray | torch.Tensor, shape (n, m)
            Tensor of training targets
        obj: 'elbo' | 'predictive'
            A string defininf the objective function to use for training. \
            It must be one of ['elbo', 'predictive].
        n_epochs: int
            Number of training epoch.
        optim_kw: Mapping[str, Any]
            A mapper of the form param_name -> param_value of optional \
            settings for the optimizer.
        verbose: bool
            An option for whether to print taining status in logger.

        Returns
        -------
            BaseExactGP
        """
        # Get defaults
        def _get_defaults() -> Mapping[str, Any]:
            """Get default settings"""
            params = {'lr': .1}
            return params

        # Force input types
        train_x, train_y = [to_tensor(t) for t in [train_x, train_y]]

        # Set training mode
        self.train()
        self.noise_gp.train()
        # self.likelihood.train()

        # Setup optimizer
        optimizer = Adam(
            self.parameters(),
            **(_get_defaults() | optim_kw)
        )

        # Set the objective function
        mll = SVSHGPVariationalELBO(self, num_data=train_y.size(0))

        # Start training loop
        # FIXME: Make this compatible with mini-batching
        for n in range(n_epochs):
            # Zero grad
            optimizer.zero_grad()

            # Call
            pred = self(train_x)
            # self.update_loss_terms(train_x)
            loss = - mll(pred, train_y, x=train_x)

            # Backward and propr
            loss.backward()
            optimizer.step()

            if n == 0 or (n + 1) % 25 == 0 and verbose:
                logger.info(
                    f'Iter {n + 1} of {n_epochs}: '
                    # f'Lenghscale: {}'
                    # f'Noise: {self.likelihood.noise.item(): .3f} - '
                    f'Loss: {loss.item(): .3f}'
                )

        # Display score on selected metrics
        display_scores(self.score(train_x, train_y))

        return self

    def predict_latent(
        self,
        test_x: np.ndarray | Tensor,
        return_ci: bool = True,
    ) -> MultivariateNormal | Tuple[MultivariateNormal, Tensor, Tensor]:
        """
        Given the test feautres, make prediction of the latent function \
            distribution and return with confidence interval.

        Parameters
        ----------
        test_x: np.ndarray | Tensor
            Input features.
        return_ci: bool
            An option for whether to return the confidence interval.

        Returns
        -------
            MultivariateNormal | Tuple[MultivariateNormal, Tensor, Tensor]
        """
        # Force input types
        test_x = to_tensor(test_x)

        # Set to eval
        self.eval()
        # self.likelihood.eval()

        with torch.no_grad():
            f_dist = self(test_x)
            lower, upper = f_dist.confidence_region()

        if return_ci:
            return f_dist, lower, upper

        return f_dist

    def predict(
        self,
        test_x: np.ndarray | Tensor,
        return_ci: bool = True,
    ) -> MultivariateNormal | Tuple[MultivariateNormal, Tensor, Tensor]:
        """
        Given the test feautres, make preduction and return the posterior \
            distribution alongside with confidence interval.

        Parameters
        ----------
        test_x: np.ndarray | Tensor
            Input features.
        return_ci: bool
            An option for whether to return the confidence interval.

        Returns
        -------
            MultivariateNormal | Tuple[MultivariateNormal, Tensor, Tensor]
        """
        # Force input types
        test_x = to_tensor(test_x)

        # Set to eval
        self.eval()
        self.noise_gp.eval()

        with torch.no_grad():
            f_dist = self(test_x)
            mean, covar = f_dist.mean, f_dist.lazy_covariance_matrix
            noise_covar = self.noise_gp.added_noise(test_x)
            y_obs = MultivariateNormal(mean, covar + noise_covar)

            lower, upper = y_obs.confidence_region()

        if return_ci:
            return f_dist, lower, upper

        return f_dist

    def score(
        self,
        test_x: np.ndarray | Tensor,
        test_y: np.ndarray | Tensor,
        methods: (
            Callable[[np.ndarray], float]
            | List[Callable[[np.ndarray], float]]
        ) = SCORES
    ) -> pd.DataFrame:
        """
        Given the test features and targets, compute the corresponding \
            predictions scores.

        Parameters
        ----------
        test_x: np.ndarray | Tensor
            Input test features.
        test_y: np.ndarray | Tensor
            Input test targets.
        methods: Callable | List[Callabe]
            The socre methods to be used.

        Returns
        -------
            DataFrame.
        """
        pred = to_numpy(self.predict(test_x, False).mean)
        actual = to_numpy(test_y)
        return compute_scores(pred, actual, methods)
