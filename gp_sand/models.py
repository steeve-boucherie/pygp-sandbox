"""
Baseline GP models
------------------
Contains generic GPs implementation to with self training (fit) \
    predit method for standardized API calls.
"""
import abc
import logging
import math
from typing import Any, Callable, List, Literal, Mapping, Tuple

# import gpytorch
from gpytorch.distributions import MultivariateNormal
# from gpytorch.gp import GP
from gpytorch.likelihoods import (
    _GaussianLikelihoodBase,
    GaussianLikelihood
)
from gpytorch.kernels import (
    ConstantKernel,
    Kernel,
    MaternKernel,
    RBFKernel,
    ScaleKernel,
)
from gpytorch.means import ConstantMean, Mean, ZeroMean
from gpytorch.mlls import (
    ExactMarginalLogLikelihood,
    KLGaussianAddedLossTerm,
    PredictiveLogLikelihood,
    VariationalELBO,
)
from gpytorch.models import ApproximateGP, ExactGP
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    VariationalStrategy,
)

from linear_operator.operators import DiagLinearOperator

import numpy as np

import pandas as pd

import torch
from torch import Tensor
from torch.optim import Adam


# PACKAGE IMPORT
# from gp_sand.likelihoods import VariationalHeteroscedasticLikelihood
from gp_sand.metrics import (
    bias,
    bias_perc,
    compute_scores,
    cov,
    display_scores,
    mae,
    nrmse,
    rmse
)
from gp_sand.utils import (
    get_inductions_points,
    is_allowed,
    to_numpy,
    to_tensor,
)


# LOGGER
logger = logging.getLogger(__name__)


# DEFAULTS
SCORES = [bias, bias_perc, cov, mae, nrmse, rmse]


# UTILS
def _add_diag(cov: Tensor, jitter: float = 1e-4) -> Tensor:
    """
    Given the input covariance matrix, add jitter to the diagonal.

    Parameters
    ----------
    cov: Tensor
        Input covariance matrix.
    jitter: float
        Magnitude of the jitter to be added.

    Returns
    -------
        Tensor
    """
    eye = torch.eye(cov.size(0), device=cov.device, dtype=cov.dtype)
    return cov + jitter * eye


def _chol_inv(cov: Tensor, jitter: float = 1e-4) -> Tensor:
    """
    Given the input covariance matrix, compute its inverse using \
        Cholesky factorization.

    Parameters
    ----------
    cov: Tensor
        Input covariance matrix.
    jitter: float
        Magnitude of the jitter to be added on the diagonal for stability.

    Returns
    -------
        Tensor
    """
    cov = _add_diag(cov)
    return torch.cholesky_inverse(torch.linalg.cholesky(cov))


# ABSTRACT
class GPInterface(abc.ABC):
    """
    Interface class for GP implementation.

    Description
    ------------
    All inheriting class must implement the following methods:
    - forward: [Tensor] -> MultivariateNormal
        Given the input features, compute the posterior distribution.
    - fit: [Tensor, Tensor] -> self
        Given the training data fit the model.
    - predict: [Tensor] -> MultivariateNormal
        Given the input features, make prediction and return also the \
        corresponding confidence region.
    """

    @abc.abstractmethod
    def forward(X: Tensor, *args, **kwargs) -> MultivariateNormal:
        """Given the input features, compute the posterior distribution."""
        raise NotImplementedError('This is an abstract class')

    @abc.abstractmethod
    def fit(X: Tensor, y: Tensor, *args, **kwargs) -> 'GPInterface':
        """Given the training data fit the model."""
        raise NotImplementedError('This is an abstract class')

    @abc.abstractmethod
    def predict(X: Tensor, *args, **kwargs) -> MultivariateNormal:
        """Given the input features, make prediction and return also the \
            corresponding confidence region."""
        raise NotImplementedError('This is an abstract class')

    @abc.abstractmethod
    def score(X: Tensor, y: Tensor, *args) -> pd.DataFrame:
        """Given the test features and target compute the corresponding \
            prediction scores."""
        raise NotImplementedError('This is an abstract class')


# EXACT GP
class BaseExactGP(ExactGP, GPInterface):
    """
    Implementation of exact GP.

    Attributes
    ----------
    train_x: np.ndarray | Tensor, shape (n, m)
        Tensor of training features.
    train_y: np.ndarray | Tensor, shape (n, m)
        Tensor of training targets.
    mean_module: Mean
        Mean function.
    covar_module: Kernel
        Covariance kernel function.
    likelihood: _GaussianLikelihoodBase
        Likelihood model.
    """

    def __init__(
        self,
        train_x: np.ndarray | Tensor | None = None,
        train_y: np.ndarray | Tensor | None = None,
        mean_module: Mean = ConstantMean(),
        covar_module: Kernel = ScaleKernel(MaternKernel(nu=2.5)),
        likelihood: _GaussianLikelihoodBase = GaussianLikelihood(),
    ):
        """Init class."""
        train_x, train_y = [to_tensor(t) for t in [train_x, train_y]]
        super().__init__(train_x, train_y, likelihood)

        # Mean and covar module
        self.mean_module = mean_module
        self.covar_module = covar_module

    # Forward
    def forward(self, X: Tensor) -> MultivariateNormal:
        """
        Given the input tensor of features, compute the resulting \
            posterior distribution.

        Parameters
        ----------
        X: Tensor
            Tensor of input features.

        Returns
        -------
            MultivariateNormal
        """
        mean_f = self.mean_module(X)
        covar_f = self.covar_module(X)
        return MultivariateNormal(mean_f, covar_f)

    # Fit/Predict
    def fit(
        self,
        train_x: np.ndarray | Tensor,
        train_y: np.ndarray | Tensor,
        n_epochs: int = 250,
        optim_kw: Mapping[str, Any] = {},
        verbose: bool = True,
    ) -> 'BaseExactGP':
        """
        Given the traning data and fitting option, fit the model.

        Parameters
        ----------
        train_x: np.ndarray | torch.Tensor, shape (n, m)
            Tensor of training features.
        train_y: np.ndarray | torch.Tensor, shape (n, m)
            Tensor of training targets
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
        self.likelihood.train()

        # Setup optimizer
        optimizer = Adam(
            self.parameters(),
            **(_get_defaults() | optim_kw)
        )

        # Set the objective function
        mll = ExactMarginalLogLikelihood(self.likelihood, self)

        # Start training loop
        for n in range(n_epochs):
            # Zero grad
            optimizer.zero_grad()

            # Call
            pred = self(train_x)
            loss = - mll(pred, train_y)

            # Backward and propr
            loss.backward()
            optimizer.step()

            if n == 0 or (n + 1) % 25 == 0 and verbose:
                logger.info(
                    f'Iter {n + 1} of {n_epochs}: '
                    # f'Lenghscale: {}'
                    f'Noise: {self.likelihood.noise.item(): .3f} - '
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
        self.likelihood.eval()

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
        self.likelihood.eval()

        with torch.no_grad():
            f_dist = self(test_x)
            y_obs = self.likelihood(f_dist)

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


# SPARSE GP
class BaseSparseGP(ApproximateGP, GPInterface):
    """
    Implementation of Sparse GP Approximation.

    Attributes
    ----------
    train_x: np.ndarray | Tensor, shape (n, m)
        Tensor of training features.
    train_y: np.ndarray | Tensor, shape (n, m)
        Tensor of training targets.
    mean_module: Mean
        Mean function.
    covar_module: Kernel
        Covariance kernel function.
    likelihood: _GaussianLikelihoodBase
        Likelihood model.
    n_points: int
        Number of inducing point for the sparse GP.
    """

    def __init__(
        self,
        train_x: np.ndarray | Tensor | None = None,
        train_y: np.ndarray | Tensor | None = None,
        mean_module: Mean = ConstantMean(),
        covar_module: Kernel = ScaleKernel(MaternKernel(nu=2.5)),
        likelihood: _GaussianLikelihoodBase = GaussianLikelihood(),
        n_points: int = 250,
    ):
        """Init class."""
        train_x, train_y = [to_tensor(t) for t in [train_x, train_y]]
        ind_points = get_inductions_points(train_x, n_points)
        var_dist = CholeskyVariationalDistribution(
            num_inducing_points=ind_points.size(0)
        )
        var_strat = VariationalStrategy(
            self,
            inducing_points=ind_points,
            variational_distribution=var_dist,
            learn_inducing_locations=True
        )
        super().__init__(var_strat)

        # Mean and covar module
        self.mean_module = mean_module
        self.covar_module = covar_module

        # Store for interval use
        self.n_points = n_points
        self.train_x = train_x
        self.train_y = train_y
        self.likelihood = likelihood

    # Forward
    def forward(self, X: Tensor) -> MultivariateNormal:
        """
        Given the input tensor of features, compute the resulting \
            posterior distribution.

        Parameters
        ----------
        X: Tensor
            Tensor of input features.

        Returns
        -------
            MultivariateNormal
        """
        mean_f = self.mean_module(X)
        covar_f = self.covar_module(X)
        return MultivariateNormal(mean_f, covar_f)

    # Fit/Predict
    def fit(
        self,
        train_x: np.ndarray | Tensor,
        train_y: np.ndarray | Tensor,
        obj: Literal['elbo', 'predictive'] = 'elbo',
        n_epochs: int = 250,
        optim_kw: Mapping[str, Any] = {},
        verbose: bool = True,
    ) -> 'BaseExactGP':
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
        self.likelihood.train()

        # Setup optimizer
        optimizer = Adam(
            self.parameters(),
            **(_get_defaults() | optim_kw)
        )

        # Set the objective function
        is_allowed(obj, ['elbo', 'predictive'])
        mll_fun = {
            'elbo': VariationalELBO,
            'predictive': PredictiveLogLikelihood
        }[obj]
        mll = mll_fun(self.likelihood, self, num_data=train_y.size(0))

        # Start training loop
        for n in range(n_epochs):
            # Zero grad
            optimizer.zero_grad()

            # Call
            pred = self(train_x)
            loss = - mll(pred, train_y)

            # Backward and propr
            loss.backward()
            optimizer.step()

            if n == 0 or (n + 1) % 25 == 0 and verbose:
                logger.info(
                    f'Iter {n + 1} of {n_epochs}: '
                    # f'Lenghscale: {}'
                    f'Noise: {self.likelihood.noise.item(): .3f} - '
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
        self.likelihood.eval()

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
        self.likelihood.eval()

        with torch.no_grad():
            f_dist = self(test_x)
            y_obs = self.likelihood(f_dist)

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


# HETEROSKEDASTIC GP
class VariationalHeteroscedasticGP(ExactGP, GPInterface):
    """
    Variational heteroskedastic GP following the implementation of \
        Lázaro-Gredilla & Titsias 2011.

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
        https://dl.acm.org/doi/10.5555/3104482.3104588

    Attributes
    ----------
    train_x: np.ndarray | Tensor, shape (n, m)
        Tensor of training features.
    train_y: np.ndarray | Tensor, shape (n, m)
        Tensor of training targets.
    covar_latent: Kernel
        Covariance module for the latent function f.
    mean_noise: Mean
        Mean function for the log-noise model.
    covar_noise: Kernel
        Covariance module for the log-noise model.
    jitter: float
        Small term to be added on the diagonal covariance for numerical \
        stability. Default is 1e-4.
    """

    def __init__(
        self,
        train_x: np.ndarray | Tensor | None = None,
        train_y: np.ndarray | Tensor | None = None,
        covar_latent: Kernel = ScaleKernel(MaternKernel(nu=2.5)),
        mean_noise: ConstantMean = ConstantMean(),
        covar_noise: Kernel = ScaleKernel(RBFKernel()) + ConstantKernel(),
        jitter: float = 1e-3,
    ):
        """Init class."""
        # Initialize super
        likelihood = GaussianLikelihood()
        likelihood.noise = 1e-4
        # likelihood = VariationalHeteroscedasticLikelihood()
        super().__init__(train_x, train_y, likelihood)
        # super().__init__(train_x, train_y, VariationalHeteroscedasticLikelihood())

        # GP for the latent function
        self.mean_latent = ZeroMean()
        self.covar_latent = covar_latent

        # GP for the log-noise
        self.mean_noise = mean_noise
        self.covar_noise = covar_noise

        # Variational parameters
        # These are the diagonal elements used to compute the \
        # mean and sigma of the variation function
        n = to_tensor(train_x).size(0)
        self.log_lamba_var = torch.nn.Parameter(
            torch.zeros(n) - 2.0  # Small values
        )

        # Misc
        self.jitter = jitter

    # Properties
    @property
    def lambda_var(self) -> torch.nn.Parameter:
        """Return the diagonal matrix Lambda."""
        return torch.diag(torch.exp(self.log_lamba_var))

    # Utils
    def compute_variational_params(self) -> Tuple[Tensor, Tensor]:
        """
        Compute the vector mean mu and covariance matrix Sigma from \
            the noise-GP prediction and the diagonal matrix lambda.

        Notes
        -----
        >>> mu = K_g(Lambda - .5 * I)1 + mu_0*1
        >>> Sigma^{-1} = K_g^{-1} + Lambda

        Where mu_0 and K_g are the mean and covariance of the noise-GP \
        prediction on the training data.

        Returns
        -------
        mu: Tensor
            Mean of the q(g)
        sigma: Tensor
            Covariance matri of q(g)
        """
        # Get noise-GP pred
        mu_g = self.mean_noise(self.train_inputs[0])
        k_g = self.covar_noise(self.train_inputs[0]).evaluate()

        # Intermediate terms
        lmbd = self.lambda_var
        n = mu_g.size(0)
        ones = torch.ones(n, device=k_g.device, dtype=k_g.dtype)

        # Compute the mean
        mu = k_g @ (_add_diag(lmbd, -0.5)) @ ones + mu_g

        # Compute the covar matrix
        k_g = _add_diag(k_g, self.jitter)
        k_g_inv = torch.inverse(k_g)
        sigma = torch.inverse(k_g_inv + lmbd)

        return mu, sigma

    def _compute_rmat(self) -> Tensor:
        """
        Compute the R-matrix based on the training data.

        Notes
        -----
        >>> r = diag(exp(mu_i - 0.5 * sigma_ii))

        Where:
        - mu and sigma are the variation distribution parameters \
            evaluated on the training data.

        Returns
        -------
            Tensor
        """
        mu, sigma = self.compute_variational_params()
        return torch.diag(torch.exp(mu - .5 * torch.diag(sigma)))

    def _compute_mvbound(self) -> MultivariateNormal:
        """
        Compute the term for the marginalized variation bound \
            estimation from the training data.
        """
        mu_f = self.mean_latent(self.train_inputs[0])
        k_f = self.covar_latent(self.train_inputs[0])
        k_f += self._compute_rmat()  # Add the R-matrix on the diagonal

        # # Compute the R term
        # mu, sigma = self.compute_variational_params()
        # k_f += torch.diag(torch.exp(mu - .5 * torch.diag(sigma)))

        # Assemble
        k_f = _add_diag(k_f, self.jitter)
        dist = MultivariateNormal(mu_f, k_f)

        return dist

    def _compute_loss_term(self) -> Tensor:
        """
        Compute the added loss terms to be included in the marginal \
            variational bounds.

        Notes
        -----
        >>> added_loss = - (.25 * trace(sigma) + KL(MVN(my, sigma) || MVN(mu_g, k_g)))

        Where:
        - mu and sigma are the variation distibution param evaluated \
            on the training inputs.
        - mu_g and k_g are the mean and covariance of the noise-GP model \
            evaluated on the training inputs.

        Returns
        -------
            Tensor
        """
        # Get the noise-GP pred
        mu_g = self.mean_noise(self.train_inputs[0])
        k_g = self.covar_noise(self.train_inputs[0]).evaluate()

        # Get variational params
        mu, sigma = self.compute_variational_params()

        # Compute the loss (trace)
        n = k_g.size(0)
        loss = .25 * torch.trace(sigma) / n

        # Add the KL-divergence
        g_dist = MultivariateNormal(mu_g, _add_diag(k_g, self.jitter))
        var_dist = MultivariateNormal(mu, _add_diag(sigma, self.jitter))
        kl_div = KLGaussianAddedLossTerm(
            var_dist,
            g_dist,
            n=n,
            data_dim=1
        )
        loss += kl_div.loss()

        return loss

    def predictive_latent(self, test_x: Tensor) -> MultivariateNormal:
        """
        Given new unseen data evaluate the posterior distribution of the \
            latent function f(x) at these points.

        Parameters
        ----------
        text_x: Tensor
            Tensor of new unseen features.

        Returns
        -------
            MultivariateNormal
        """
        # Get the terms
        k_f = self.covar_latent(self.train_inputs[0]).evaluate()
        k_f = _add_diag(k_f, self.jitter) + self._compute_rmat()
        # k_f_inv = torch.cholesky_inverse(torch.linalg.cholesky(k_f))
        alpha = torch.linalg.solve(k_f, self.train_targets)

        self.eval()
        with torch.no_grad():
            k_f_star = self.covar_latent(test_x, self.train_inputs[0]).evaluate()
            k_f_star_star = self.covar_latent(test_x).evaluate()

            # Get the mean
            mean_f = k_f_star @ alpha

            # Get the covar
            covar_f = k_f_star_star - k_f_star @ torch.linalg.solve(k_f, k_f_star.t())

        return MultivariateNormal(mean_f, _add_diag(covar_f))

    def predictive_noise(self, test_x: Tensor) -> DiagLinearOperator:
        """
        Given new unseen data evaluate the noise, g(x), to be added to the \
            latent function, f(x), at these points.

        Description
        -----------
        Predictions are multivariate normal with mean mu_star:
        >>> mu_star = k_g_star @ (lambda - .5 * I) + mu_0

        and covar
        >>> sigma_star = k_g_star_star - k_g_star^T @ (k_g + lambda^{-1})^{-1} @ k_g_star

        Parameters
        ----------
        text_x: Tensor
            Tensor of new unseen features.

        Returns
        -------
            Tensor
        """
        # Evaluate on the training data
        lmbd = self.lambda_var
        # mu_g = self.mean_noise(self.train_inputs[0])
        mu_0 = self.mean_noise(torch.ones(1)).item()
        k_g = self.covar_noise(self.train_inputs[0]).evaluate()

        self.eval()

        # Compute  the terms
        k_g_star = self.covar_noise(test_x, self.train_inputs[0]).evaluate()
        k_g_star_star = self.covar_noise(test_x).evaluate()

        # Get the mean, mu_star
        n = k_g.size(0)
        eye = torch.eye(n, device=k_g.device, dtype=k_g.dtype)
        ones = torch.ones(n)
        mu_star = k_g_star @ (lmbd - .5 * eye) @ ones + mu_0

        # Get the covar sigma_star
        sigma_star = k_g_star_star
        sigma_star -= k_g_star @ torch.linalg.solve(k_g + _chol_inv(lmbd), k_g_star.t())

        noise_diag = torch.exp(mu_star + .5 * torch.diag(sigma_star))

        # return noise_diag
        return DiagLinearOperator(noise_diag)

    # Forward
    def forward(self, X: Tensor) -> MultivariateNormal:
        """
        Given the input tensor of features, compute the resulting \
            posterior distribution.

        Parameters
        ----------
        X: Tensor
            Tensor of input features.

        Returns
        -------
            MultivariateNormal
        """
        if self.training:
            # Get only the term for MV bound estimate
            return self._compute_mvbound()

        else:
            # Get the posterior predictive
            return self.predictive_latent(X)

    # Fit/Predict
    def pre_train(
        self,
        train_x: np.ndarray | Tensor,
        train_y: np.ndarray | Tensor,
        n_epochs: int = 150,
        optim_kw: Mapping[str, Any] = {},
        verbose: bool = True,
    ) -> None:
        """
        Given the training data, pre-train the latent and noise functions using \
            an homoscedastic versions of the model.

        Parameters
        ----------
        train_x: np.ndarray | torch.Tensor, shape (n, m)
            Tensor of training features.
        train_y: np.ndarray | torch.Tensor, shape (n, m)
            Tensor of training targets
        n_epochs: int
            Number of training epoch.
        optim_kw: Mapping[str, Any]
            A mapper of the form param_name -> param_value of optional \
            settings for the optimizer.
        verbose: bool
            An option for whether to print taining status in logger.

        Returns
        -------
            None
        """
        # Setup base GP
        logger.info('Pre-train latent GP (homoscedastic).')
        base_gp = BaseExactGP(
            self.train_inputs[0],
            self.train_targets,
            mean_module=self.mean_latent,
            covar_module=self.covar_latent,
            likelihood=GaussianLikelihood(),
        )

        # Train latent
        train_kw = dict(
            n_epochs=n_epochs,
            optim_kw=optim_kw,
            verbose=verbose
        )
        base_gp.fit(train_x, train_y, **train_kw)

        # Setup and train noise GP
        logger.info('Initliaze the noise GP parameters.')

        # Use the homoscedastic noise value for the mean
        with torch.no_grad():
            self.mean_noise.constant.fill_(
                math.log(base_gp.likelihood.noise.item())
            )

    def fit(
        self,
        train_x: np.ndarray | Tensor,
        train_y: np.ndarray | Tensor,
        n_epochs: int = 150,
        optim_kw: Mapping[str, Any] = {},
        pre_train: bool = True,
        pre_train_kw: Mapping[str, Any] = {},
        verbose: bool = True,
    ) -> 'BaseExactGP':
        """
        Given the traning data and fitting option, fit the model.

        Parameters
        ----------
        train_x: np.ndarray | torch.Tensor, shape (n, m)
            Tensor of training features.
        train_y: np.ndarray | torch.Tensor, shape (n, m)
            Tensor of training targets
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

        if pre_train:
            logger.info('Pre-training (homoscedastic) latent and noise models.')
            self.pre_train(train_x, train_y, **pre_train_kw)

        # Set training mode
        self.train()
        self.likelihood.eval()

        # Setup optimizer
        optimizer = Adam(
            [
                {'params': self.mean_latent.parameters()},
                {'params': self.covar_latent.parameters()},
                {'params': self.mean_noise.parameters()},
                {'params': self.covar_noise.parameters()},
                {'params': [self.log_lamba_var]},
            ],
            **(_get_defaults() | optim_kw)
        )

        # Set the objective function
        mll = ExactMarginalLogLikelihood(self.likelihood, self)

        # Start training loop
        for n in range(n_epochs):
            # Zero grad
            optimizer.zero_grad()

            # Call
            pred = self(train_x)
            loss = - (mll(pred, train_y) - self._compute_loss_term())

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
        self.likelihood.eval()

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
        self.likelihood.eval()

        with torch.no_grad():
            f_dist = self(test_x)
            mean, covar = f_dist.mean, f_dist.lazy_covariance_matrix
            noise_covar = self.predictive_noise(test_x)
            y_obs = f_dist.__class__(mean, covar + noise_covar)

            lower, upper = y_obs.confidence_region()

        if return_ci:
            return y_obs, lower, upper

        return y_obs

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
