"""Mixture of Expert GP"""
import logging
from copy import deepcopy
from typing import Any, List, Mapping, Tuple

from gp_sand.utils import to_tensor

from gpytorch.constraints import Interval, Positive
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import Kernel
from gpytorch.likelihoods import _GaussianLikelihoodBase
from gpytorch.means import Mean
from gpytorch.mlls import MarginalLogLikelihood, SumMarginalLogLikelihood
from gpytorch.models import ApproximateGP, GP
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    VariationalStrategy
)

import numpy as np

import torch
from torch import Tensor
from torch.distributions import Normal, MixtureSameFamily, Categorical
from torch.optim import Adam


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


# HELPER CLASS
class ExpertSVGP(ApproximateGP):
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
        var_dist= CholeskyVariationalDistribution(
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


class MoEVariationalELBO(MarginalLogLikelihood):
    """
    Custom ELBO for a Mixture of SVGP Experts.
    Mirrors the structure of GPyTorch's VariationalELBO but handles
    a mixture likelihood and per-expert KL terms.

    Parameters
    ----------
    likelihoods : list of GaussianLikelihood
        One per expert.
    model : MoSVGP
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
        model: 'MixtureofExpertSVGP',
        num_data: int, 
        beta: float = 1.0,
    ):
        # Register with the first likelihood as the "main" one
        # for compatibility with GPyTorch's MLL interface
        super().__init__(likelihoods[0], model)
        self.likelihoods = likelihoods
        self.num_data = num_data
        self.beta = beta

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

        mix  = Categorical(probs=weights.t())             # (N, K)
        comp = Normal(means.t(), variances.sqrt().t())        # (N, K)
        gmm  = MixtureSameFamily(mix, comp)
        log_lik = gmm.log_prob(targets).sum() / outputs[0].event_shape[0]     # scalar

        # Get the KL divergence to force the variational distribution
        # toward the real one.
        # Scale by num_data/batch_size to match GPyTorch convention
        batch_size = targets.shape[0]
        kl_scale = self.beta / self.num_data

        kl = sum(
            expert.variational_strategy.kl_divergence().sum()
            for expert in self.model.experts
        )

        # Assemble the ELBO term
        elbo = log_lik - kl_scale * kl

        return elbo

# MAIN CLASS
class MixtureofExpertSVGP(GP):
    """Mixture of Expert combining prediction from individual SVGP \
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
        self.experts: List[ExpertSVGP] = [
            ExpertSVGP(
                ind_points[k],
                means[k],
                covars[k],
            )
            for k in range(n_experts)
        ]
        self.likelihoods = likelihoods

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
    def get_gates(self, x: Tensor) -> List[Tensor]:
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
        weights  = self.get_gates(x)          # (K, N)
        latents  = self.forward(x)        # list of K MVN

        obs_means = torch.stack(
            [lik.marginal(lat).mean
             for lik, lat in zip(self.likelihoods, latents)], dim=0
        )  # (K, N)

        obs_vars = torch.stack(
            [lik.marginal(lat).variance
             for lik, lat in zip(self.likelihoods, latents)], dim=0
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
        n_epochs: int = 150,
        optim_kw: Mapping[str, Any] = {},
        verbose: bool = True,
    ) -> 'MixtureofExpertSVGP':
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
        [model.train() for model in self.experts]
        [likelihood.train() for likelihood in self.likelihoods]

        # Setup optimizer
        all_params = [
            {'params': self.parameters()},
        ]
        for expert, llk in zip(self.experts, self.likelihoods):
            all_params += [
                {'params': expert.parameters()},
                {'params': llk.parameters()}
            ]

        optimizer = Adam(
            all_params,
            **(_get_defaults() | optim_kw)
        )

        # Set the objective function
        mll = MoEVariationalELBO(self.likelihoods, self, num_data=train_y.size(0))

        # Start training loop
        for n in range(n_epochs):
            # Zero grad
            optimizer.zero_grad()

            # Call
            predictions = self(train_x)
            loss = -mll(predictions, train_y, x=train_x)

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

        # # Display score on selected metrics
        # display_scores(self.score(train_x, train_y))

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

        # Set to eval mode
        self.eval()
        [model.eval() for model in self.experts]
        [likelihood.eval() for likelihood in self.likelihoods]

        with torch.no_grad():
            mean, var, _ = self.mixture_distribution(test_x)
            std = var.sqrt()

            lower = mean - 1.96 * std
            upper = mean + 1.96 * std

        if return_ci:
            return mean, lower, upper

        return mean


