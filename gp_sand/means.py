"""Customized mean modules for GPs regression."""
import logging
from typing import Any, Mapping

from gpytorch.means import Mean

import numpy as np

import torch
from torch import nn, Tensor
from torch.optim import Adam


# PACKAGE IMPORT
from gp_sand.utils import to_tensor


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

    # Fit/Predict
    def fit(
        self,
        train_x: np.ndarray | Tensor,
        train_y: np.ndarray | Tensor,
        n_epochs: int = 250,
        optim_kw: Mapping[str, Any] = {},
        verbose: bool = True,
    ) -> 'HyperbolicTangentMean':
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

        # Setup optimizer
        optimizer = Adam(
            self.parameters(),
            **(_get_defaults() | optim_kw)
        )

        # Set the objective function
        obj = nn.MSELoss()
        optimizer = Adam(
            self.parameters(),
            **(_get_defaults() | optim_kw)
        )

        # Start training loop
        for n in range(n_epochs):
            # Zero grad
            optimizer.zero_grad()

            # Call
            pred = self(train_x)
            loss = obj(pred, train_y)

            # Backward and propr
            loss.backward()
            optimizer.step()

            if n == 0 or (n + 1) % 25 == 0 and verbose:
                logger.info(
                    f'Iter {n + 1} of {n_epochs}: '
                    f'Loss: {loss.item(): .3f}'
                )

        # # Display score on selected metrics
        # display_scores(self.score(train_x, train_y))

        return self

    def predict(
        self,
        test_x: np.ndarray | Tensor,
        return_ci: bool = True,
    ) -> Tensor:
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

        with torch.no_grad():
            pred = self(test_x)

        return pred


class LogisticMean(Mean):
    """
    Paramettric mean function to use in GP,
        approximating it as an logistic function.

    The power curve is modelled as followed
    >>> p(x) = scale / (1 + exp(-shape * (x - loc)))

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
        denom = 1 + torch.exp(-1 * self.shape * (x - self.loc))
        power = self.scale / denom
        return power.squeeze()

    # Fit/Predict
    def fit(
        self,
        train_x: np.ndarray | Tensor,
        train_y: np.ndarray | Tensor,
        n_epochs: int = 250,
        optim_kw: Mapping[str, Any] = {},
        verbose: bool = True,
    ) -> 'LogisticMean':
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

        # Setup optimizer
        optimizer = Adam(
            self.parameters(),
            **(_get_defaults() | optim_kw)
        )

        # Set the objective function
        obj = nn.MSELoss()
        optimizer = Adam(
            self.parameters(),
            **(_get_defaults() | optim_kw)
        )

        # Start training loop
        for n in range(n_epochs):
            # Zero grad
            optimizer.zero_grad()

            # Call
            pred = self(train_x)
            loss = obj(pred, train_y)

            # Backward and propr
            loss.backward()
            optimizer.step()

            if n == 0 or (n + 1) % 25 == 0 and verbose:
                logger.info(
                    f'Iter {n + 1} of {n_epochs}: '
                    f'Loss: {obj.item(): .3f}'
                )

        # # Display score on selected metrics
        # display_scores(self.score(train_x, train_y))

        return self

    def predict(
        self,
        test_x: np.ndarray | Tensor,
        return_ci: bool = True,
    ) -> Tensor:
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

        with torch.no_grad():
            pred = self(test_x)

        return pred
