"""Customized mean modules for GPs regression."""
import abc
import logging
from typing import Any, Mapping

from gpytorch.constraints import Interval, Positive
from gpytorch.means import Mean

import numpy as np

import torch
from torch import nn, Tensor
from torch.optim import Adam


# PACKAGE IMPORT
from gp_sand.utils import to_tensor


# LOGGER
logger = logging.getLogger(__name__)


# ABSTRACT
class MeanInterface(abc.ABC):
    """
    Interface class for Mean Module implementation.

    Description
    ------------
    All inheriting class must implement the following methods:
    - forward: [Tensor] -> Tensor
        Given the input features, compute the mean values.
    - fit: [Tensor, Tensor] -> self
        Given the training data fit the model.
    - predict: [Tensor] -> Tensor
        Given the input features, return predictions.
    """

    @abc.abstractmethod
    def forward(X: Tensor, *args, **kwargs) -> Tensor:
        """Given the input features, compute the posterior distribution."""
        raise NotImplementedError('This is an abstract class')

    @abc.abstractmethod
    def fit(X: Tensor, y: Tensor, *args, **kwargs) -> 'MeanInterface':
        """Given the training data fit the model."""
        raise NotImplementedError('This is an abstract class')

    @abc.abstractmethod
    def predict(X: Tensor, *args, **kwargs) -> Tensor:
        """Given the input features, make prediction and return also the \
            corresponding confidence region."""
        raise NotImplementedError('This is an abstract class')

    # @abc.abstractmethod
    # def score(X: Tensor, y: Tensor, *args) -> pd.DataFrame:
    #     """Given the test features and target compute the corresponding \
    #         prediction scores."""
    #     raise NotImplementedError('This is an abstract class')


# CUSTOM MEAN MODULE
class HyperbolicTangentMean(Mean, MeanInterface):
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


class LogisticMean(Mean, MeanInterface):
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


class PiecewiseLinearMean(Mean, MeanInterface):
    """
    Smooth piecewise mean function for wind turbine pitch modelling.

    For normalised feature x in [0, 1]:
      - Low speed  (x << x_t): m(x) ≈ mu0          (constant pitch)
      - High speed (x >> x_t): m(x) ≈ mu0 + beta*(x - x_t)  (linear ramp)
      - Transition is smoothed by a sigmoid gate controlled by 'sharpness'

    Notes
    -----
    Works with the normalized torque, generator speed or power as feature.

    Parameters
    ----------
    mu_1: float
        Values of the pitch int he constant-pithc regime (1) \
        constainted to [mu_1_min, mu_1_max]
    x_01: float
        Transition point, between startup (0) and constant regimes (1) \
        constrained to [x_01_min, x_01_max]
    x_12: float
        Transition point, between variable speed (1) close-to-rated (2) regimes
        constrained to [x_12_min, x_12_max]
    beta_0: float
        Ramp slope in the startup regime (0) \
        Constrained to be positive.
    beta_2: float
        Ramp slope in the close-to-rated regime (2) \
        Constrained to be positive.
    sharpness: float
        Sigmoid steepness, constrained to Positive
    """

    def __init__(
        self,
        mu_1_min: float = -2.5,
        mu_1_max: float = 2.5,
        x_01_min: float = 0.05,
        x_01_max: float = 0.35,
        x_01_init: float = 0.2,
        beta_0_init: float = 5.,
        x_12_min: float = 0.70,
        x_12_max: float = 0.95,
        x_12_init: float = 0.85,
        beta_2_init: float = 5.,
        sharpness_init: float = 50.0,
    ):
        """
        Initialize class object.

        Parameters
        ----------
        mu_1_min: float
            Lower bound of the interval constraining the baseline pitch value.
        mu_1_max: float
            Upper bound of the interval constraining the baseline pitch value.
        x_01_min: float
            Lower bound of the interval constraining the transition point from \
            startup (0) to constant pitch (1) regimes.
        x_01_max: float
            Upper bound of the interval constraining the transition point from \
            startup (0) to constant pitch (1) regimes.
        x_01_init: float
            Intial vale of the transition point from startup (0) to constant \
            pitch (1) regimes.
        beta_0_init: float
            Initial values of the slope in the startup regime (0).
        x_12_min: float
            Lower bound of the interval constraining the transition from constant \
            pitch (1) to close-to-rated (2) regimes.
        x_12_max: float
            Upper bound of the interval constraining the transition from constant \
            pitch (1) to close-to-rated (2) regimes.
        x_12_init: float
            Intial vale of the transition point from constant pitch (1) to \
            close-to-rated (2) regimes.
        beta_2_init: float
            Initial values of the slope in the close-to-rated regimes (2).
        sharpness_init: float
            Initial value for the parameter controlling the sharpness of transition \
            between regimes.
        """
        super().__init__()

        # Baseline pitch values
        # Constrained in [mu_min, mu_max] interval
        mu_1_constraint = Interval(mu_1_min, mu_1_max)
        self.register_parameter(
            name='raw_mu_1',
            parameter=torch.nn.Parameter(
                mu_1_constraint.inverse_transform(
                    torch.tensor(0.)
                )
            )
        )
        self.register_constraint('raw_mu_1', mu_1_constraint)

        # Transition points between startup and constant pitch regimes
        # Constrained in [x_01_min, x_01_max] interval
        x_01_constraint = Interval(x_01_min, x_01_max)
        self.register_parameter(
            name='raw_x_01',
            parameter=torch.nn.Parameter(
                x_01_constraint.inverse_transform(
                    torch.tensor(x_01_init)
                )
            )
        )
        self.register_constraint('raw_x_01', x_01_constraint)

        # Slope in the startup regime
        # Constraint to be POSTIVE (with a negative sign in the equation)
        beta_0_constraint = Positive()
        self.register_parameter(
            name='beta_0',
            parameter=torch.nn.Parameter(
                beta_0_constraint.inverse_transform(
                    torch.tensor(beta_0_init)
                )
            )
        )
        self.register_constraint('beta_0', beta_0_constraint)

        # Transition points between constant pitch close-to-rated regimes
        # Constrained in [x_12_min, x_12_max] interval
        x_12_constraint = Interval(x_12_min, x_12_max)
        self.register_parameter(
            name='raw_x_12',
            parameter=torch.nn.Parameter(
                x_12_constraint.inverse_transform(
                    torch.tensor(x_12_init)
                )
            )
        )
        self.register_constraint('raw_x_12', x_12_constraint)

        # Slope in the close-to-rated regime
        # Constraint to be POSTIVE (with a negative sign in the equation)
        beta_2_constraint = Positive()
        self.register_parameter(
            name='beta_2',
            parameter=torch.nn.Parameter(
                beta_2_constraint.inverse_transform(
                    torch.tensor(beta_2_init)
                )
            )
        )
        self.register_constraint('beta_2', beta_2_constraint)

        # Transition stiffness
        # Constrained to be positive
        sharpness_constraint = Positive()
        self.register_parameter(
            name='raw_sharpness',
            parameter=torch.nn.Parameter(
                sharpness_constraint.inverse_transform(
                    torch.tensor(sharpness_init)
                )
            )
        )
        self.register_constraint('raw_sharpness', sharpness_constraint)

    # Properties
    @property
    def mu_1(self):
        return self.raw_mu_1_constraint.transform(self.raw_mu_1)

    @property
    def x_01(self):
        return self.raw_x_01_constraint.transform(self.raw_x_01)

    @property
    def beta_0(self):
        return self.raw_beta_0_constraint.transform(self.raw_beta_0)

    @property
    def x_12(self):
        return self.raw_x_12_constraint.transform(self.raw_x_12)

    @property
    def beta_2(self):
        return self.raw_beta_2_constraint.transform(self.raw_beta_2)

    @property
    def sharpness(self):
        return self.raw_sharpness_constraint.transform(self.raw_sharpness)

    # Methods
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Squeeze to 1-D if input is (N, 1)
        if x.dim() == 2 and x.shape[-1] == 1:
            x = x.squeeze(-1)

        # Sigmoid gate 01: 1 for low speeds, 0 for medium to high speeds
        gate_0 = torch.sigmoid(self.sharpness * (x - self.x_01))

        # Sigmoid gate 12: 1 for high speeds, 0 for low to medium speeds
        gate_1 = torch.sigmoid(self.sharpness * (x - self.x_12))

        # Piecewise mean
        mean = self.mu_1 \
            + gate_1 * self.beta_2 * (x - self.x_12) \
            - gate_0 * self.beta_0 * (x - self.x_01)

        return mean

    # Fit/Predict
    def fit(
        self,
        train_x: np.ndarray | Tensor,
        train_y: np.ndarray | Tensor,
        n_epochs: int = 250,
        optim_kw: Mapping[str, Any] = {},
        verbose: bool = True,
    ) -> 'PiecewiseLinearMean':
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
        obj = torch.nn.MSELoss()
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
