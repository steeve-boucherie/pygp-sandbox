"""Mixture of Expert GP"""
import logging
import sys
from copy import deepcopy
from typing import Any, Callable, List, Literal, Mapping, Tuple

from gpytorch.constraints import Interval, Positive
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import Kernel
from gpytorch.likelihoods import _GaussianLikelihoodBase
from gpytorch.means import Mean
from gpytorch.mlls import MarginalLogLikelihood
from gpytorch.mlls._approximate_mll import _ApproximateMarginalLogLikelihood
from gpytorch.models import ApproximateGP, GP, IndependentModelList
from gpytorch.utils.generic import length_safe_zip
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    VariationalStrategy
)

import numpy as np

import pandas as pd

import torch
from torch import nn, Tensor
from torch.distributions import Normal, MixtureSameFamily, Categorical
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset


# PACKAGE IMPORTS
from gp_sand.metrics import (
    compute_scores,
    # display_scores
)
from gp_sand.models import GPInterface, SCORES
from gp_sand.utils import to_numpy, to_tensor


# LOGGER
logger = logging.getLogger(__name__)


# UTILS
def _make_list(elem: Any, n_elems: int) -> List[Any]:
    """
    Given an element test if it is a list return a list of n copies \
        if otherwise.

    Parameters
    ----------
    elem: Any
        Element to be tested.
    n_elems: int
        Number of copies to generate.
    """
    if isinstance(elem, list):
        return elem

    return [deepcopy(elem) for _ in range(n_elems)]


def _verify_length(elements: List[Any], req_length: int) -> None:
    """
    Given a list of elements, verify it match the required length, \
        and raise a ValueError if otherwise.

    Parameters
    ----------
    elements: List[Any]
        Input list to be tested.
    req_length: int
        The required number of elements.

    Raises
    ------
        ValueError
    """
    if len(elements) != req_length:
        msg = f'The number of elements must correspond to the required length : {req_length}'
        logger.error(msg)
        raise ValueError(msg)


# HELPER CLASS
class ExpertVSGP(ApproximateGP):
    """GP model for the individual expert to be \
        used in the Mixture of Experts."""

    # Init
    def __init__(
        self,
        ind_points: np.ndarray | Tensor,
        mean_module: Mean,
        covar_module: Kernel,
    ):
        """Init class"""
        ind_points = to_tensor(ind_points)
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

        # Mean and covar
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


class MoEVariationalELBO(_ApproximateMarginalLogLikelihood):
    """
    Custom ELBO for a Mixture of SVGP Experts.
    Mirrors the structure of GPyTorch's VariationalELBO but handles
    a mixture likelihood and per-expert KL terms.

    Parameters
    ----------
    likelihoods : list of GaussianLikelihood
        One per expert.
    model : MoEVSGP
        The mixture model exposing .experts and .gates(x).
    num_data : int
        Total training set size (for correct KL scaling).
    beta : float
        KL weight. Default 1.0 (standard ELBO).
        Can be annealed during training.
    """

    def __init__(
        self,
        likelihoods: List[_GaussianLikelihoodBase],
        model: 'MixtureofExpertVSGP',
        num_data: int,
        beta: float = 1.0,
    ):
        # Register with the first likelihood as the "main" one
        # for compatibility with GPyTorch's MLL interface
        super().__init__(likelihoods[0], model, num_data, beta, True)
        self.likelihoods = likelihoods
        # self.num_data = num_data
        # self.beta = beta

    def _log_likelihood_term(
        self,
        outputs: List[MultivariateNormal],
        targets: Tensor,
        **kwargs
    ) -> Tensor:
        """
        Compute the mixture's log-likelihood term of the ELBO.

        Attributes
        ----------
        outputs : List[MultivariateNormal]
            The individual expert predictions
        targets : Tensor (N,)
            Observed values.
        kwargs: Mapping[str, Any]
            Optional argument it must constain the tensor of training features \
            to estimate the mixture's weights.
        """
        # Get the gates
        weights = self.model.get_gates(kwargs['x'])  # (K, N)

        # Get the log-probability of the mixture
        means = torch.stack(
            [
                lik.marginal(out).mean
                for lik, out in zip(self.likelihoods, outputs)
            ],
            dim=0
        )  # (K, N)

        variances = torch.stack(
            [
                lik.marginal(out).variance
                for lik, out in zip(self.likelihoods, outputs)
            ],
            dim=0
        )  # (K, N) — already includes per-expert noise

        mix = Categorical(probs=weights.t())             # (N, K)
        comp = Normal(means.t(), variances.sqrt().t())        # (N, K)
        gmm = MixtureSameFamily(mix, comp)
        log_lik = gmm.log_prob(targets).sum()  # scalar

        return log_lik

    def forward(
        self,
        outputs: List[MultivariateNormal],
        targets: Tensor,
        **kwargs
    ) -> Tensor:
        """
        Parameters
        ----------
        outputs : List[MultivariateNormal]
            The individual expert predictions
        targets : Tensor (N,)
            Observed values.

        Returns
        -------
        ELBO: Tensor
            The estimated ELBO
        """
        # Get the likelihood term
        num_batch = outputs[0].event_shape[0]
        log_likelihood = (
            self._log_likelihood_term(
                outputs,
                targets,
                **kwargs
            )
            .div(num_batch)
        )

        # Get the KL divergence to force the variational distribution
        # toward the real one.
        kl_scale = self.beta / self.num_data

        kl = sum(
            expert.variational_strategy.kl_divergence()  # .sum()
            for expert in self.model.experts
        )

        # Assemble the ELBO term
        elbo = log_likelihood - kl_scale * kl

        return elbo


class SumMoEVariationalELBO(MarginalLogLikelihood):
    """Sum of marginal MoE Variation ELBO, to be used with Multi-Output models.

    Args:
        likelihood: A MultiOutputLikelihood
        model: A MultiOutputModel
        mll_cls: The Marginal Log Likelihood class (default: ExactMarginalLogLikelihood)

    In case the model outputs are independent, this provives the MLL of the multi-output model.

    """

    def __init__(
        self,
        likelihood: _GaussianLikelihoodBase,  # There for compatibility not used
        model: 'MoEVSGPList',
        num_data: List[int],
        beta: float | List[float] = 1.0,
    ):
        super().__init__(model.likelihood, model)
        # NOTE: This hack is to by-pass the init of Super that does not allow
        # to pass kwargs for the selected MLL model while MoEVariationalELBO
        # requires the num_data parameters.
        _verify_length(num_data, model.n_models)
        beta = _make_list(beta, model.n_models)
        _verify_length(beta, model.n_models)
        self.mlls = nn.ModuleList([
            MoEVariationalELBO(model.likelihood, model, num_data=nd, beta=b)
            for model, nd, b in zip(model.models, num_data, beta)
        ])

    def forward(
        self,
        outputs: List[MultivariateNormal],
        targets: List[Tensor],
        params: List[Mapping[str, Any]] = []
    ) -> Tensor:
        """
        Execute forward step.

        Parameters
        ----------
        outputs: List[MultivariateNormal]
            List of lists  of multivariate normals return by the model list.
        targets: List[Tensor]
            List of training targets.
        params: List[Mapping[str, Any]]
            List of optional parameters to pass to each ELBO MLL.

        Returns
        -------
            Tensor
        """
        if len(params) == 0:
            sum_mll = sum(
                mll(*output, *target)
                for mll, output, target in length_safe_zip(self.mlls, outputs, targets)
            )
        else:
            sum_mll = sum(
                mll(output, target, **iparams)
                for mll, output, target, iparams in length_safe_zip(
                    self.mlls, outputs, targets, params)
            )
        return sum_mll.div_(len(self.mlls))


# MAIN CLASS
class MixtureofExpertVSGP(GP, GPInterface):
    """Mixture of Expert combining prediction from individual VSGP \
        and a simple gating mechanism."""

    # Init
    def __init__(
        self,
        n_experts: int,
        ind_points: np.ndarray | Tensor | List[np.ndarray | Tensor],
        means: Mean | List[Mean],
        covars: Kernel | List[Kernel],
        likelihoods: _GaussianLikelihoodBase | List[_GaussianLikelihoodBase],
        transition_points: List[float],
        tp_constraints: List[Interval] | None = None,
        sharpness: float = 30.,
    ):
        super().__init__()
        self.n_experts = n_experts

        # Validate the induction points
        ind_points = _make_list(ind_points, n_experts)
        ind_points = [to_tensor(_ind) for _ind in ind_points]
        _verify_length(ind_points, n_experts)

        # Validate the mean models
        means = _make_list(means, n_experts)
        _verify_length(means, n_experts)

        # Validate the covar models
        covars = _make_list(covars, n_experts)
        _verify_length(covars, n_experts)

        # Validate the covar models
        likelihoods = _make_list(likelihoods, n_experts)
        _verify_length(likelihoods, n_experts)

        # Set up the experts and likelihoods
        self.experts: nn.ModuleList = nn.ModuleList([
            ExpertVSGP(
                ind_points[k],
                means[k],
                covars[k],
            )
            for k in range(n_experts)
        ])
        self.likelihood = nn.ModuleList(likelihoods)

        # Gating
        sharpness_constraint = Positive()
        self.register_parameter(
            'sharpness_raw',
            torch.nn.Parameter(
                sharpness_constraint.inverse_transform(
                    torch.tensor(sharpness)
                )
            )
        )
        self.register_constraint('sharpness_raw', sharpness_constraint)

        _verify_length(transition_points, n_experts - 1)
        transition_points.sort()
        for n, pt in enumerate(transition_points):
            self.register_parameter(
                f'x_{n + 1}',
                torch.nn.Parameter(torch.tensor(pt))
            )

        if tp_constraints is not None:
            _verify_length(tp_constraints, n_experts - 1)
            for n, const in enumerate(tp_constraints):
                self.register_constraint(f'x_{n + 1}', const)

    # Properties
    @property
    def name(self) -> str:
        """Return the class name."""
        return self.__class__.__name__

    @property
    def n_transition_pts(self) -> int:
        """Return the number of transition points."""
        return self.n_experts - 1

    @property
    def means(self) -> List[Mean]:
        """Return the list of mean models."""
        return [model.mean_module for model in self.experts]

    @property
    def covars(self) -> List[Mean]:
        """Return the list of mean models."""
        return [model.covar_module for model in self.experts]

    @property
    def sharpness(self) -> Tensor:
        """Return the constrained sharpness"""
        return self.sharpness_raw_constraint.transform(self.sharpness_raw)

    @property
    def transition_points(self) -> List[torch.nn.Parameter]:
        """
        Get the values of the nth transition point.

        Returns
        -------
            List[torch.nn.Parameter]
        """
        transition_points = []
        for n in range(self.n_transition_pts):
            name = f'x_{n + 1}'
            x_t = self.__getattr__(name)
            cons_name = f'{name}_constraint'
            if hasattr(self, cons_name):
                x_t = self.__getattr__(cons_name).transform(x_t)

            transition_points.append(x_t)

        return transition_points

    # Util
    def get_gates(self, x: Tensor) -> Tensor:
        """Given the inputs compute the gate values."""
        if x.ndim == 2 and x.size(1) == 1:
            x = x.squeeze()

        gates = []
        for n, x_t in enumerate(self.transition_points):
            w = 1 - torch.sigmoid(self.sharpness * (x - x_t))
            if n != 0:
                w = w - sum(gates[k] for k in range(len(gates)))
            gates.append(w)

        gates.append(
            torch.ones_like(gates[0]) - sum(gates[k]for k in range(len(gates)))
        )
        gates = torch.stack(gates, axis=0)

        return gates

    def mixture_distribution(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Given the inputs values, returns the aggregated mixture \
        as (mean, var) used for predictions.

        Parameters
        ----------
        x: Tensor
            Tensor of input features.

        Returns
        -------
        mixture_mean: Tensor
            The estimated means of the mixture model
        mixture_var: Tensor
            The estimated variance of mixture model
        weights: Tensor
            The model weights estimated at each inputs.
        """
        weights = self.get_gates(x)          # (K, N)
        latents = self.forward(x)        # list of K MVN

        obs_means = torch.stack(
            [lik.marginal(lat).mean
             for lik, lat in zip(self.likelihood, latents)], dim=0
        )  # (K, N)

        obs_vars = torch.stack(
            [lik.marginal(lat).variance
             for lik, lat in zip(self.likelihood, latents)], dim=0
        )  # (K, N)

        mixture_mean = (weights * obs_means).sum(dim=0)
        expected_var = (weights * obs_vars).sum(dim=0)
        variance_of_means = (
            weights * (obs_means - mixture_mean.unsqueeze(0)).pow(2)
        ).sum(dim=0)

        return mixture_mean, expected_var + variance_of_means, weights

    # Forward
    def forward(self, x: Tensor) -> MultivariateNormal:
        """Forward method."""
        return [model(x) for model in self.experts]

    # Fit/Predict
    def fit(
        self,
        train_x: np.ndarray | Tensor,
        train_y: np.ndarray | Tensor,
        n_epochs: int = 10,
        batch_size: int = 256,
        batch_kw: Mapping[str, Any] = {},
        optim_kw: Mapping[str, Any] = {},
        verbose: bool = True,
    ) -> 'MixtureofExpertVSGP':
        """
        Given the traning data and fitting option, fit the model.

        Parameters
        ----------
        train_x: np.ndarray | torch.Tensor, shape (n, m)
            Tensor of training features.
        train_y: np.ndarray | torch.Tensor, shape (n, m)
            Tensor of training targets
        n_epochs: int
            Number of training epochs.
        batch_size: int
            Batch size to split the training data in mini-batches.
        batch_kw: Mapping[str, Any]
            A mapper of th eofrm param_name -> param_value of optional \
            settings to pass to the DataLoader for mini-batching.
        optim_kw: Mapping[str, Any]
            A mapper of the form param_name -> param_value of optional \
            settings for the optimizer.
        verbose: bool
            An option for whether to print taining status in logger.

        Returns
        -------
            MixtureofExpertSVGP
        """
        # Get defaults
        def _get_defaults(obj: Literal['batch', 'optim']) -> Mapping[str, Any]:
            """Get default settings"""
            params = {
                'batch': {
                    'batch_size': batch_size,
                    'shuffle': True,
                    'drop_last': True
                },
                'optim': {'lr': .1}
            }[obj]
            return params

        # Force input types
        train_x, train_y = [to_tensor(t) for t in [train_x, train_y]]

        # Set training mode
        self.train()
        self.likelihood.train()

        optimizer = Adam(
            self.parameters(),
            **(_get_defaults('optim') | optim_kw)
        )

        # Set the objective function
        mll = MoEVariationalELBO(self.likelihood, self, num_data=train_y.size(0))

        # Create data loader for mini-batching
        data_loader = DataLoader(
            TensorDataset(train_x, train_y),
            **(_get_defaults('batch') | batch_kw)
        )

        # Start training loop
        for n in range(n_epochs):
            for batch_x, batch_y in data_loader:
                # Zero grad
                optimizer.zero_grad()

                # Call
                pred = self(batch_x)
                # self.update_loss_terms(train_x)
                loss = - mll(pred, batch_y, x=batch_x)

                # Backward and propr
                loss.backward()
                optimizer.step()

            # if n == 0 or (n + 1) % 25 == 0 and verbose:
            if verbose:
                msg = (
                    f'{self.name} - Iter {n + 1} of {n_epochs}: '
                    f'Loss: {loss.item(): .3f}'
                )
                # weights = self.get_gates(train_x).mean(dim=1)
                # msg += ' - '.join([
                #     f' weight expert #{n:02d} = {round(weights[0].item(), 2)}'
                #     for n in range(self.n_experts)
                # ])
                # msg += '\n'
                # msg += ' - '.join([
                #     f' noise expert #{n:02d} = {lkh.noise.item():.3e}'
                #     for n, lkh in enumerate(self.likelihoods)
                # ])
                logger.info(msg)

        # Display score on selected metrics
        # display_scores(self.score(train_x, train_y))

        return self

    def predict(
        self,
        test_x: np.ndarray | Tensor,
        return_ci: bool = True,
        ci_width: float = 1.96
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
        ci_width: float
            The width of the confidence interval in multiple of the standard \
            deviation at each point. Default is 1.96 (90% confidence interval).

        Returns
        -------
            MultivariateNormal | Tuple[MultivariateNormal, Tensor, Tensor]
        """
        # Force input types
        test_x = to_tensor(test_x)

        # Set to eval mode
        self.eval()
        # [model.eval() for model in self.experts]
        # [likelihood.eval() for likelihood in self.likelihood]

        with torch.no_grad():
            mean, var, _ = self.mixture_distribution(test_x)
            std = var.sqrt()

            lower = mean - ci_width * std
            upper = mean + ci_width * std

        if return_ci:
            return mean, lower, upper

        return mean

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


# CHILD
class IterativeTrimmingMofEtVSGP(MixtureofExpertVSGP):
    """
    Implementation of the Mixture of Experts (MoE) using Sparse Variational \
        Gaussian Processes (SVGP) with iterative trimming of the data to \
        filter outliers.

    Notes
    -----
    The class leverage all the methods of the MixtureofExpertSVGP. The only \
    difference concerns the `fit` method that includes the outlier removal steps.

    See: # TODO: Add reference
    """

    # Properties
    @property
    def name(self) -> str:
        """Return the class name."""
        return self.__class__.__name__

    # Fit/Predict
    def fit(
        self,
        train_x: np.ndarray | Tensor,
        train_y: np.ndarray | Tensor,
        n_iter: int = 5,
        n_epochs: int = 100,
        thresh_sd: float = 2.62,
        thresh_max: float | None = None,
        optim_kw: Mapping[str, Any] = {},
        verbose: bool = True,
    ) -> 'IterativeTrimmingMofEtVSGP':
        """
        Given the traning data and fitting option, fit the model.

        Parameters
        ----------
        train_x: np.ndarray | torch.Tensor, shape (n, m)
            Tensor of training features.
        train_y: np.ndarray | torch.Tensor, shape (n, m)
            Tensor of training targets.
        n_iter: int
            The number of training iteration.
        n_epochs: int
            Number of training epoch for each iteration.
        thresh_sd: float
            The threshold for outlier detection in mulitple of the standard deviation.
        thresh_max: float | None
            (Optional) Max absolute threshold used for capping the standard deviation \
            threshold in high-variance regions.
        optim_kw: Mapping[str, Any]
            A mapper of the form param_name -> param_value of optional \
            settings for the optimizer.
        verbose: bool
            An option for whether to print taining status in logger.

        Returns
        -------
            BaseExactGP
        """
        for n in range(n_iter):
            logger.info(f'{self.name} - Start training iteration {n + 1} (of {n_iter}).')

            # Run training of the MoE
            logger.info(f'{self.name} - MoE training step.')
            super().fit(
                train_x,
                train_y,
                n_epochs=n_epochs,
                optim_kw=optim_kw,
                verbose=verbose,
            )

            # Run training of the MoE
            logger.info(f'{self.name} - Outlier removal step.')
            with torch.no_grad():
                mean, var, _ = self.mixture_distribution(train_x)
                upper = thresh_sd * var.sqrt()

                # Apply thresholding
                limit = [sys.maxsize, thresh_max][thresh_max is not None]
                upper = torch.where(
                    upper <= limit,
                    upper,
                    limit
                )

                res = train_y - mean
                ind = res <= upper

            logger.info(
                f'{self.name} - Rejecting {len(ind) - ind.numpy().sum()} training samples '
                f'({(100 * (1 - ind.numpy().mean())).round(2)}% of the data).'
            )
            train_x = train_x[ind, :]
            train_y = train_y[ind]

        return self


# MODEL LIST
class MoEVSGPList(IndependentModelList, GPInterface):
    """
    Independent list of Mixture of Expert (MoE) using Sparse Variational \
        Gaussian Processes for convienent model fitting/predicting.

    Notes
    -----
    This is a wrapper around GPyTorch `IndependentModelList` with a custom \
        training function to handle the variational ELBO.

    Attributes
    ----------
    models: List[MixtureofExpertSVGP]
        The list of Mixture of Expert SVGPs.
    """

    def __init__(self, *models: List[MixtureofExpertVSGP]):
        super().__init__(*models)

    # Properties
    @property
    def name(self) -> str:
        """Return the class name."""
        return self.__class__.__name__

    @property
    def n_models(self) -> int:
        """Return the number of models making the model list."""
        return len(self.models)

    # Utils
    def validate_inputs(self, inputs: List[np.ndarray | Tensor]) -> None:
        """
        Given the list of inputs (feature or target), verify it matches with number \
            of model and raise ValueError if otherwise.

        Parameters
        ----------
        inputs: List[np.ndarray | Tensor]
            Inputs to be tested.

        Raises
        ------
            ValueError
        """
        len_inputs = len(inputs)
        if len_inputs != self.n_models:
            msg = f'Input length ({len_inputs}) does not match the number of models ' \
                  f'in the list ({self.n_models}). Please check your inputs.'
            logger.error(msg)
            raise ValueError(msg)

    # Fit/Predict
    def fit(
        self,
        train_inputs: List[np.ndarray | Tensor],
        train_targets: List[np.ndarray | Tensor],
        n_epochs: int = 150,
        optim_kw: Mapping[str, Any] = {},
        verbose: bool = True,
    ) -> 'MoEVSGPList':
        """
        Given the traning data and fitting option, fit the model.

        Parameters
        ----------
        train_x: List[np.ndarray | torch.Tensor]
            List of tensors of training features.
        train_y: List[np.ndarray | torch.Tensor]
            List of tensors of training features.
        n_epochs: int
            Number of training epoch.
        optim_kw: Mapping[str, Any]
            A mapper of the form param_name -> param_value of optional \
            settings for the optimizer.
        verbose: bool
            An option for whether to print taining status in logger.

        Returns
        -------
            MoESVGPList
        """
        # Get defaults
        def _get_defaults() -> Mapping[str, Any]:
            """Get default settings"""
            params = {'lr': .1}
            return params

        # Force input types
        # TODO: Allow for mini-batching
        train_inputs = [to_tensor(t) for t in train_inputs]
        train_targets = [to_tensor(t) for t in train_targets]

        # Set training mode
        self.train()
        self.likelihood.train()

        optimizer = Adam(
            [{'params': self.parameters()},],
            **(_get_defaults() | optim_kw)
        )

        # Set the objective function
        mll = SumMoEVariationalELBO(
            self.likelihood,
            self,
            [train_y.size(0) for train_y in train_targets]
        )

        # Start training loop
        for n in range(n_epochs):
            # Zero grad
            optimizer.zero_grad()

            # Call
            predictions = self(*train_inputs)
            loss = -mll(
                predictions,
                train_targets,
                params=[{'x': train_x} for train_x in train_inputs]
            )

            # Backward and propr
            loss.backward()
            optimizer.step()

            if n == 0 or (n + 1) % 25 == 0 and verbose:
                msg = (
                    f'{self.name} - Iter {n + 1} of {n_epochs}: '
                    f'Loss: {loss.item(): .3f}'
                )
                # weights = self.get_gates(train_x).mean(dim=1)
                # msg += ' - '.join([
                #     f' weight expert #{n:02d} = {round(weights[0].item(), 2)}'
                #     for n in range(self.n_experts)
                # ])
                # msg += '\n'
                # msg += ' - '.join([
                #     f' noise expert #{n:02d} = {lkh.noise.item():.3e}'
                #     for n, lkh in enumerate(self.likelihoods)
                # ])
                logger.info(msg)

        # Display score on selected metrics
        # display_scores(self.score(train_x, train_y))

        return self

    def predict(
        self,
        test_inputs: np.ndarray | Tensor | List[np.ndarray | Tensor],
        return_ci: bool = True,
        ci_width: float = 1.96
    ) -> List[MultivariateNormal] | List[Tuple[MultivariateNormal, Tensor, Tensor]]:
        """
        Given the test features, make prediction and return the posterior \
            distribution alongside with confidence interval.

        Parameters
        ----------
        test_x: np.ndarray | Tensor | List[np.ndarray | Tensor]
            Input features.
        return_ci: bool
            An option for whether to return the confidence interval.
        ci_width: float
            The width of the confidence interval in multiple of the standard \
            deviation at each point. Default is 1.96 (90% confidence interval).

        Returns
        -------
            MultivariateNormal | Tuple[MultivariateNormal, Tensor, Tensor]
        """
        # Force input types
        _make_list(test_inputs, n_elems=self.n_models)
        test_inputs = [to_tensor(test_x) for test_x in test_inputs]

        # Set to eval mode
        self.eval()
        # [model.eval() for model in self.experts]
        # [likelihood.eval() for likelihood in self.likelihood]

        outputs = []
        for model, test_x in length_safe_zip(self.models, test_inputs):
            outputs.append(model.predict(test_x, return_ci, ci_width))

        return outputs

    def score(
        self,
        test_inputs: List[np.ndarray | Tensor],
        test_targets: List[np.ndarray | Tensor],
        methods: (
            Callable[[np.ndarray], float]
            | List[Callable[[np.ndarray], float]]
        ) = SCORES
    ) -> pd.DataFrame:
        """Given the test features and target compute the corresponding \
            prediction scores."""
        scores = [
            model.score(test_x, test_y)
            for model, test_x, test_y in length_safe_zip(
                self.models,
                test_inputs,
                test_targets
            )
        ]

        return scores


class ITMoEVSGPList(MoEVSGPList):
    """
    Independent list of Iterative Trimming Mixture of Expert (MoE) using Sparse \
        Variational Gaussian Processes for convienent model fitting/predicting.

    Notes
    -----
    This is a wrapper around GPyTorch `IndependentModelList` with a custom \
        training function to handle the variational ELBO.

    Attributes
    ----------
    models: List[MixtureofExpertSVGP]
        The list of Mixture of Expert SVGPs.
    """

    # Fit/Predict
    def fit(
        self,
        train_inputs: List[np.ndarray | Tensor],
        train_targets: List[np.ndarray | Tensor],
        n_iter: int = 5,
        n_epochs: int = 100,
        thresh_sd: float = 2.62,
        thresh_max: float | None = None,
        optim_kw: Mapping[str, Any] = {},
        verbose: bool = True,
    ) -> 'ITMoEVSGPList':
        """
        Given the traning data and fitting option, fit the model.

        Parameters
        ----------
        train_x: List[np.ndarray | torch.Tensor]
            List of tensors of training features.
        train_y: List[np.ndarray | torch.Tensor]
            List of tensors of training features.
        n_epochs: int
            Number of training epoch.
        optim_kw: Mapping[str, Any]
            A mapper of the form param_name -> param_value of optional \
            settings for the optimizer.
        verbose: bool
            An option for whether to print taining status in logger.

        Returns
        -------
            MoESVGPList
        """
        for n in range(n_iter):
            logger.info(f'{self.name} - Start training iteration {n + 1} (of {n_iter}).')

            # Run training of the MoE
            logger.info(f'{self.name} - MoE training step.')
            super().fit(
                train_inputs,
                train_targets,
                n_epochs=n_epochs,
                optim_kw=optim_kw,
                verbose=verbose,
            )

            # Run training of the MoE
            logger.info(f'{self.name} - Outlier removal step.')
            for m, model in enumerate(self.models):
                train_x = train_inputs[m]
                train_y = train_targets[m]
                with torch.no_grad():
                    mean, var, _ = model.mixture_distribution(train_x)
                    upper = thresh_sd * var.sqrt()

                    # Apply thresholding
                    limit = [sys.maxsize, thresh_max][thresh_max is not None]
                    upper = torch.where(
                        upper <= limit,
                        upper,
                        limit
                    )

                    res = train_y - mean
                    ind = res <= upper

                logger.info(
                    f'{self.name} - Rejecting {-1 * (~ind.numpy().sum())} training samples '
                    f'({(100 * (1 - ind.numpy().mean())).round(2)}% of the data).'
                )
                train_inputs[m] = train_x[ind, :]
                train_targets[m] = train_y[ind]

        return self
