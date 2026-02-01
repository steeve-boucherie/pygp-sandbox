"""
Baseline GP models
------------------
Contains generic GPs implementation to with self training (fit) \
    predit method for standardized API calls.
"""
import abc
import logging
from typing import Any, Callable, List, Literal, Mapping, Tuple

# import gpytorch
from gpytorch.distributions import MultivariateNormal
# from gpytorch.gp import GP
from gpytorch.likelihoods import (
    _GaussianLikelihoodBase,
    GaussianLikelihood
)
from gpytorch.kernels import Kernel, MaternKernel, ScaleKernel
from gpytorch.means import ConstantMean, Mean
from gpytorch.mlls import (
    ExactMarginalLogLikelihood,
    PredictiveLogLikelihood,
    VariationalELBO,
)
from gpytorch.models import ApproximateGP, ExactGP
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    VariationalStrategy,
)

import numpy as np

import pandas as pd

import torch
from torch import Tensor
from torch.optim import Adam


# PACKAGE IMPORT
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
        train_x: np.ndarray | Tensor,
        train_y: np.ndarray | Tensor,
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
        train_x: np.ndarray | Tensor,
        train_y: np.ndarray | Tensor,
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
