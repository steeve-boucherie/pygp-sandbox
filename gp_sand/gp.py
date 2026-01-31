"""Wrapper classes for fitting and making prediction with \
    Gaussian Process"""
import logging
from typing import Any, Literal, Mapping

import gpytorch
from gpytorch.likelihoods import _GaussianLikelihoodBase, GaussianLikelihood
from gpytorch.mlls import PredictiveLogLikelihood, VariationalELBO
from gpytorch.mlls._approximate_mll import _ApproximateMarginalLogLikelihood
from gpytorch.models import ApproximateGP
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    VariationalStrategy
)

import numpy as np

# from pydant

from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.utils import resample
# from sklearn.preprocessing import MinMaxScaler

import torch
from torch import Tensor
from torch.optim import Optimizer


# LOGGER
logger = logging.getLogger(__name__)


# UTILS
def get_inductions_points(
    x_feat: np.ndarray | Tensor,
    n_points: int = 150,
) -> Tensor:
    """
    Given the input features, compute the initial locations \
        of the inductions points.

    Parameters
    ----------
    x_feat: np.ndarray | Tensor
        Input array of features.
    n_points: int
        The number of induction points to generate for each variables

    Returns
    -------
        Tensor
    """
    if isinstance(x_feat, Tensor):
        x_feat = x_feat.numpy()

    ind_points = np.percentile(
        x_feat,
        np.linspace(0, 100, n_points),
        axis=0
    )

    x_feat = np.atleast_2d(x_feat)
    np.random.random((n_points, x_feat.shape[1]))

    return torch.tensor(ind_points).to(torch.float32)


# CLASS
class _SparseGP(ApproximateGP):
    """
    Base class for Sparse GP

    Attributes
    ----------
    inducing_points: Tensor
        A tensor of the shape (n_pts, n_feat) defining the initial \
        location of the inductions point.
    preprocessor: TransformerMixin | Pipeline | None
        An instance of sklearn transformer or pipeline to apply preprocessing \
        to the data prior to model fitting.
    """

    def __init__(
        self,
        inducing_points: Tensor,
        preprocessor: Pipeline | TransformerMixin | None = None,
    ):
        """Instantiate class object."""
        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=inducing_points.size(0),
        )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points=inducing_points,
            variational_distribution=variational_distribution,
            learn_inducing_locations=True
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class SparseHSGPRegressor():
    """
    Sparse Gaussian-Process regressor with heteroskedastic noise.

    Attributes
    ----------
    n_pts: int
        The number of inducing points.
    likelihood: GaussianLikelihood
        Likelihood model
    preprocessor: TransformerMixin | Pipeline | None
        An instance of sklearn transformer or pipeline to apply preprocessing \
        to the data prior to model fitting.
    optimize: 'Adam'
        The optimizer to use for fitting the model. Only Adam allowed for now.
    objective: 'elbo', 'pred_ll'
        The objective function.
    n_epochs: int
        Number of epoch to use while training the model.
    """

    _OPTIM = ['Adam']
    _OBJECTIVE = ['elbo', 'pred_ll']

    def __init__(
        self,
        n_pts: int,
        likelihood: _GaussianLikelihoodBase = GaussianLikelihood(),
        preprocessor: Pipeline | TransformerMixin | None = None,
        optimizer: Literal['Adam'] = 'Adam',
        objective: Literal['elbo', 'pred_ll'] = 'pred_ll',
        n_epochs: int = 250,
    ):
        """Init class object"""

        self.n_pts = n_pts
        self.likelihood = likelihood
        self.preprocessor = preprocessor
        self.n_epochs = n_epochs

        if optimizer not in self._OPTIM:
            msg = 'Invalid value for attribute "optimizer".\n It must be one of: ' \
                  f'{self._OPTIM},\n but received: {optimizer}.\n' \
                  'Please check your inputs.'
            logger.error(msg)
            raise ValueError(msg)

        self._optim = optimizer

        if objective not in self._OBJECTIVE:
            msg = 'Invalid value for attribute "objective".\n It must be one of: ' \
                  f'{self._OBJECTIVE},\n but received: {objective}.\n' \
                  'Please check your inputs.'
            logger.error(msg)
            raise ValueError(msg)

        self._obj = objective

        # Internal
        self.model: _SparseGP = None

    # Utils
    def _get_optim(self) -> Optimizer:
        """Get optimizer class (not init)"""
        optim = {'Adam': torch.optim.Adam}[self._optim]
        return optim

    def _get_obj(self) -> _ApproximateMarginalLogLikelihood:
        """Get the training objective (not init)"""
        obj = {
            'elbo': VariationalELBO,
            'pred_ll': PredictiveLogLikelihood,
        }[self._obj]

        return obj

    # Methods
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SparseHSGPRegressor':
        """
        Given the input features and target, fit the model parameters.

        Parameters
        ----------
        X: np.ndarray
            Input array of features.
        y: np.ndarray
            Input array of targets.
        """
        # Preprocess input
        if self.preprocessor is not None:
            X = self.preprocessor.fit_transform(X)

        # Setup model (if needed)
        if self.model is None:
            self.model = _SparseGP(get_inductions_points(X, self.n_pts))

        # Set models in training model
        self.model.train()
        self.likelihood.train()

        # Get optimizer and ojbective functions
        optimizer = self._get_optim()(
            list(self.model.parameters()) + list(self.likelihood.parameters()),
            lr=0.1
        )
        obj = self._get_obj()(
            self.likelihood,
            self.model,
            num_data=y.shape[0]
        )

        # Force input types
        X = torch.tensor(X).to(torch.float32)
        y = torch.tensor(y).to(torch.float32)

        # Start training
        for n in range(self.n_epochs):
            # Set 0-grad
            optimizer.zero_grad()

            # Get the output
            pred = self.model(X)

            # Loass and backward prop
            loss = - obj(pred, y)
            loss.backward()
            optimizer.step()

            # Logger
            if (n + 1) % 50 == 0:
                logger.info(f'Iter {n + 1} of {self.n_epochs} - Loss: {loss.item()}')

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make Prediction with the fitter model.

        Parameters
        ----------
        X: np.ndarray
            Input array of unobserved data.

        Returns
        -------
            np.ndarray
        """
        if self.model is None:
            msg = 'Fit model first!'
            logger.error(msg)
            raise ValueError(msg)

        if self.preprocessor is not None:
            X = self.preprocessor.transform(X)

        # Force input types
        X = torch.tensor(X).to(torch.float32)

        with torch.no_grad():
            f_dist = self.model(X)

        return f_dist.mean.numpy()

    # Classmethods
    @classmethod
    def from_dict(cls, param_dict: Mapping[str, Any]) -> 'SparseHSGPRegressor':
        """
        Instantiate class object from the dictionnary of input settings.

        Parameters
        ----------
        param_dict: Dict[str, Any]
            Dictionnary of input settings.

        Returns
        -------
            SparseHSGPRegressor
        """
        # # Check inputs
        # allowed_keys = set(cls.__annotations__.keys())
        # input_keys = set(param_dict.keys())

        # diff = input_keys - allowed_keys
        # if diff != set({}):
        #     msg = 'Some inputs are not valid attribute for class' \
        #           f'"{cls.__name__}".' f'keys must be a subset of ' \
        #           f'"{allowed_keys}", but received extra keys "{diff}". '\
        #           'Please check your inputs.'
        #     logger.error(msg)
        #     raise KeyError(msg)

        return cls(**param_dict)


class BatchedSparseHSGPRegressor():
    """
    Sparse Gaussian-Process regressor with heteroskedastic noise using \
        Bootstrapping batches for training

    Attributes
    ----------
    n_pts: int
        The number of inducing points.
    likelihood: GaussianLikelihood
        Likelihood model
    preprocessor: TransformerMixin | Pipeline | None
        An instance of sklearn transformer or pipeline to apply preprocessing \
        to the data prior to model fitting.
    optimize: 'Adam'
        The optimizer to use for fitting the model. Only Adam allowed for now.
    objective: 'elbo', 'pred_ll'
        The objective function.
    n_batches: int
        Number of training batches to generate.
    batch_size: float
        The batch size in fraction of the total training set size.
    random_seed: int
        Seed number for reproducibility of the results
    """

    _OPTIM = ['Adam']
    _OBJECTIVE = ['elbo', 'pred_ll']

    def __init__(
        self,
        n_pts: int,
        likelihood: _GaussianLikelihoodBase = GaussianLikelihood(),
        preprocessor: Pipeline | TransformerMixin | None = None,
        optimizer: Literal['Adam'] = 'Adam',
        objective: Literal['elbo', 'pred_ll'] = 'pred_ll',
        n_batches: int = 100,
        batch_size: float = 0.8,
        random_seed: int | None = None
    ):
        """Init class object"""

        self.n_pts = n_pts
        self.likelihood = likelihood
        self.preprocessor = preprocessor
        self.n_batches = n_batches
        self.batch_size = batch_size
        self.random_seed = random_seed

        if optimizer not in self._OPTIM:
            msg = 'Invalid value for attribute "optimizer".\n It must be one of: ' \
                  f'{self._OPTIM},\n but received: {optimizer}.\n' \
                  'Please check your inputs.'
            logger.error(msg)
            raise ValueError(msg)

        self._optim = optimizer

        if objective not in self._OBJECTIVE:
            msg = 'Invalid value for attribute "objective".\n It must be one of: ' \
                  f'{self._OBJECTIVE},\n but received: {objective}.\n' \
                  'Please check your inputs.'
            logger.error(msg)
            raise ValueError(msg)

        self._obj = objective

        # Internal
        self.model: _SparseGP = None

    # Utils
    def _get_optim(self) -> Optimizer:
        """Get optimizer class (not init)"""
        optim = {'Adam': torch.optim.Adam}[self._optim]
        return optim

    def _get_obj(self) -> _ApproximateMarginalLogLikelihood:
        """Get the training objective (not init)"""
        obj = {
            'elbo': VariationalELBO,
            'pred_ll': PredictiveLogLikelihood,
        }[self._obj]

        return obj

    # Methods
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SparseHSGPRegressor':
        """
        Given the input features and target, fit the model parameters.

        Parameters
        ----------
        X: np.ndarray
            Input array of features.
        y: np.ndarray
            Input array of targets.
        """
        # Preprocess input
        if self.preprocessor is not None:
            X = self.preprocessor.fit_transform(X)

        # Generate the training batches
        logger.info('Generate training batch')
        n_train = int(np.ceil(self.batch_size * y.shape[0]))
        Xs, ys = [], []
        rng = np.random.RandomState(self.random_seed)
        for _ in range(self.n_batches):
            _X, _y = resample(
                *[X, y],
                n_samples=n_train,
                stratify=y,
                random_state=rng
            )
            Xs.append(torch.tensor(_X).to(torch.float32))
            ys.append(torch.tensor(_y).to(torch.float32))

        # Setup model (if needed)
        if self.model is None:
            self.model = _SparseGP(get_inductions_points(X, self.n_pts))

        # Set models in training model
        self.model.train()
        self.likelihood.train()

        # Get optimizer and ojbective functions
        optimizer = self._get_optim()(
            list(self.model.parameters()) + list(self.likelihood.parameters()),
            lr=0.1
        )
        obj = self._get_obj()(
            self.likelihood,
            self.model,
            num_data=y.shape[0]
        )

        # Force input types
        X = torch.tensor(X).to(torch.float32)
        y = torch.tensor(y).to(torch.float32)

        # Start training
        logger.info('Start training.')
        for n, (_X, _y) in enumerate(zip(Xs, ys)):
            # Set 0-grad
            optimizer.zero_grad()
            # Get the output
            pred = self.model(_X)
            # Loss and backward prop
            loss = - obj(pred, _y)
            loss.backward()
            optimizer.step()

            # Logger
            if (n + 1) % 50 == 0:
                logger.info(f'Iter {n + 1} of {self.n_batches} - Loss: {loss.item()}')

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make Prediction with the fitter model.

        Parameters
        ----------
        X: np.ndarray
            Input array of unobserved data.

        Returns
        -------
            np.ndarray
        """
        if self.model is None:
            msg = 'Fit model first!'
            logger.error(msg)
            raise ValueError(msg)

        if self.preprocessor is not None:
            X = self.preprocessor.transform(X)

        # Force input types
        X = torch.tensor(X).to(torch.float32)

        with torch.no_grad():
            f_dist = self.model(X)

        return f_dist.mean.numpy()

    # Classmethods
    @classmethod
    def from_dict(cls, param_dict: Mapping[str, Any]) -> 'SparseHSGPRegressor':
        """
        Instantiate class object from the dictionnary of input settings.

        Parameters
        ----------
        param_dict: Dict[str, Any]
            Dictionnary of input settings.

        Returns
        -------
            SparseHSGPRegressor
        """
        # # Check inputs
        # allowed_keys = set(cls.__annotations__.keys())
        # input_keys = set(param_dict.keys())

        # diff = input_keys - allowed_keys
        # if diff != set({}):
        #     msg = 'Some inputs are not valid attribute for class' \
        #           f'"{cls.__name__}".' f'keys must be a subset of ' \
        #           f'"{allowed_keys}", but received extra keys "{diff}". '\
        #           'Please check your inputs.'
        #     logger.error(msg)
        #     raise KeyError(msg)

        return cls(**param_dict)
