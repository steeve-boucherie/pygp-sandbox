"""
Implemention of Variational Sparse Heteroskedastic Gaussian Process \
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

__all__ = [
    'HeteroskedasticLikelihood',
    'SparseGP',
]

import logging
import math
import warnings
from typing import Any, Callable, List, Mapping, Tuple

from gpytorch.distributions import base_distributions, MultivariateNormal
from gpytorch.kernels import Kernel, RBFKernel, ScaleKernel
from gpytorch.likelihoods import Likelihood
from gpytorch.means import ConstantMean, Mean, ZeroMean
from gpytorch.mlls import AddedLossTerm
from gpytorch.models import ApproximateGP, GP
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


# PACKAGE IMPORTS
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
from gp_sand.models import GPInterface
from gp_sand.utils import get_inductions_points, to_numpy, to_tensor


# LOGGER
logger = logging.getLogger(__name__)


# DEFAULTS
SCORES = [bias, bias_perc, cov, mae, nrmse, rmse]


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


# ADDED LOSS TERMS
class KLDivergenceLoss(AddedLossTerm):
    """
    KL divergence term for the VSHGP ELBO (Liu et al. 2020, Eq. 9):

        KL(q(g_u) || p(g_u))

    where:
        q(g_u) = N(mu_u, Sigma_u)  -- derived from Lambda_nn via Eq. 11
        p(g_u) = N(mu_0 * 1, K_uu) -- GP prior on inducing values of g

    This term is subtracted from the ELBO during training.

    Notes
    -----
    Requires a forward pass through the parent GP before calling `loss`,
    as it relies on cached mu_u and Sigma_u.

    Parameters
    ----------
    gp_g : ReparametrizedVariationalGP
        The noise GP instance that owns log_lambda_var and exposes
        the cached inducing distributions via `get_inducing_distributions`.
    """

    def __init__(self, gp_g: 'ReparametrizedVariationalGP'):
        super().__init__()
        self.gp_g = gp_g

    def loss(self, *args, **kwargs) -> Tensor:
        """
        Compute KL(q(g_u) || p(g_u)).

        Returns
        -------
        Tensor
            Scalar KL divergence.

        Raises
        ------
        RuntimeError
            If called before a forward pass has populated the cache.
        """
        q_gu, p_gu = self.gp_g.get_inducing_distributions()
        return kl_divergence(q_gu, p_gu)


class TraceGLoss(AddedLossTerm):
    """
    Trace regularisation term on q(g) for the VSHGP ELBO (Liu et al. 2020, Eq. 9):

        0.25 * Tr[Sigma_g]

    where Sigma_g is the posterior covariance of g at training points (Eq. 10b).
    Only the diagonal is used, avoiding O(n^2) memory.

    This term is subtracted from the ELBO during training.

    Notes
    -----
    Requires a forward pass through the parent GP before calling `loss`,
    as it relies on the cached Sigma_g diagonal.

    Parameters
    ----------
    gp_g : ReparametrizedVariationalGP
        The noise GP instance that exposes the cached `sigma_g_diag`.
    """

    def __init__(self, gp_g: 'ReparametrizedVariationalGP'):
        super().__init__()
        self.gp_g = gp_g

    def loss(self, *args, **kwargs) -> Tensor:
        """
        Compute 0.25 * Tr[Sigma_g].

        Returns
        -------
        Tensor
            Scalar trace term.

        Raises
        ------
        RuntimeError
            If called before a forward pass has populated the cache.
        """
        return 0.25 * self.gp_g.sigma_g_diag.sum()


# HELPER CLASSES
# TODO: Remove since not Needed, make the VSHGP directly inheriting from
# the ApproximateGP class
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


# REPARAMETRIZED VARIATIONAL GP FOR NOISE
class ReparametrizedVariationalGP(ApproximateGP):
    """
    Variational sparse GP for the log-noise function g in VSHGP,
    using the reparametrization of Liu et al. (2020), Section II.C.

    Instead of treating q(g_u) = N(mu_u, Sigma_u) as a free variational
    distribution (as in standard CholeskyVariationalDistribution), this class
    reparametrizes q(g_u) through a diagonal matrix Lambda_nn (one scalar per
    training point), following Eq. 11:

        mu_u     = K_un (Lambda - 0.5*I) 1 + mu_0 * 1          (Eq. 11a)
        Sigma_u^{-1} = K_uu^{-1} + Omega_nu^T Lambda Omega_nu   (Eq. 11b)

    The posterior q(g) at any input is then computed via Eq. 10:

        mu_g    = Omega_nu (mu_u - mu_0*1) + mu_0*1             (Eq. 10a)
        Sigma_g = K_nn - Q_nn + Omega_nu Sigma_u Omega_nu^T     (Eq. 10b)

    Benefits of the reparametrization (vs free q(g_u)):
        - Reduces variational parameters from O(u^2) to n scalars
        - Constrains the search space (Lambda entries are non-negative)
        - Initialisation at Lambda = 0.5*I sets q(g_u) = p(g_u) (prior)

    The KL divergence KL(q(g_u) || p(g_u)) and the trace regularisation
    0.25 * Tr[Sigma_g] are registered as `AddedLossTerm` instances and
    computed from cached intermediates after each forward pass.

    Notes
    -----
    - The internal CholeskyVariationalDistribution is used only as a
      placeholder to satisfy ApproximateGP's constructor. Its parameters
      are excluded from the optimiser in the parent VSHGP model.
    - log_lambda_var is initialised lazily on the first forward call,
      once the training set size n is known.
    - Only the diagonal of Sigma_g is computed (Eq. 10b), avoiding O(n^2)
      memory. This is sufficient for all ELBO terms.

    Parameters
    ----------
    inducing_points : Tensor, shape (u, d)
        Initial inducing point locations for g.
    covar_module : Kernel
        Covariance kernel k^g. Typically ScaleKernel(RBFKernel(ard_num_dims=d)).
    mean_module : Mean
        Mean function for g. Should be ConstantMean to learn mu_0.
    learn_inducing_locations : bool
        Whether to optimise inducing point locations jointly. Default True.

    Attributes
    ----------
    log_lambda_var : nn.Parameter, shape (n,)
        Log-space variational parameters. Initialised at log(0.5).
        Exponentiated to give the diagonal of Lambda_nn.
    covar_module : Kernel
        Kernel for g.
    mean_module : Mean
        Mean for g (learns mu_0 via ConstantMean).

    References
    ----------
    Liu et al. (2020), arXiv:1811.01179v3, Section II.C, Eq. 10-11.
    """

    def __init__(
        self,
        inducing_points: Tensor,
        covar_module: Kernel,
        mean_module: Mean,
        learn_inducing_locations: bool = True,
    ):
        # Placeholder variational distribution — not used directly.
        # Its parameters are excluded from the optimiser in VSHGP.fit.
        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=inducing_points.size(0)
        )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=learn_inducing_locations,
        )
        super().__init__(variational_strategy)

        self.mean_module = mean_module
        self.covar_module = covar_module

        # Variational parameter — initialised lazily in forward()
        # once training set size n is known.
        # Shape: (n,), stored in log-space to enforce non-negativity.
        # Initialisation at log(0.5) → Lambda = 0.5*I → q(g_u) = p(g_u).
        self.log_lambda_var: torch.nn.Parameter | None = None

        # Cache for AddedLossTerms — populated by forward()
        self._mu_u: Tensor | None = None
        self._Sigma_u: Tensor | None = None
        self._sigma_g_diag: Tensor | None = None

        # Register additional ELBO loss terms
        self.register_added_loss_term('kl_g', KLDivergenceLoss(self))
        self.register_added_loss_term('trace_g', TraceGLoss(self))

    # -------------------------------------------------------------------------
    # Properties — expose cached quantities to AddedLossTerms
    # -------------------------------------------------------------------------

    @property
    def sigma_g_diag(self) -> Tensor:
        """
        Diagonal of Sigma_g at training points (Eq. 10b).

        Shape: (n,)

        Raises
        ------
        RuntimeError
            If accessed before a forward pass.
        """
        if self._sigma_g_diag is None:
            raise RuntimeError(
                "sigma_g_diag is not available. "
                "Call forward() on training data first."
            )
        return self._sigma_g_diag

    # -------------------------------------------------------------------------
    # Utils
    # -------------------------------------------------------------------------

    def get_inducing_distributions(
        self,
    ) -> Tuple[torch.distributions.MultivariateNormal,
               torch.distributions.MultivariateNormal]:
        """
        Return q(g_u) and p(g_u) for KL divergence computation.

        q(g_u) = N(mu_u, Sigma_u)       -- from reparametrization (Eq. 11)
        p(g_u) = N(mu_0 * 1, K_uu)      -- GP prior on inducing values

        Returns
        -------
        q_gu : torch.distributions.MultivariateNormal
        p_gu : torch.distributions.MultivariateNormal

        Raises
        ------
        RuntimeError
            If accessed before a forward pass.
        """
        if self._mu_u is None or self._Sigma_u is None:
            raise RuntimeError(
                "Inducing distributions are not available. "
                "Call forward() on training data first."
            )

        X_u = self.variational_strategy.inducing_points         # (u, d)
        K_uu = self.covar_module(X_u).to_dense()                # (u, u)
        mu_0 = self.mean_module.constant                         # scalar

        q_gu = torch.distributions.MultivariateNormal(
            self._mu_u,
            self._Sigma_u
        )
        p_gu = torch.distributions.MultivariateNormal(
            mu_0 * torch.ones(
                X_u.size(0),
                device=X_u.device,
                dtype=X_u.dtype
            ),
            K_uu
        )
        return q_gu, p_gu

    def _init_lambda(self, n: int, device: torch.device, dtype: torch.dtype) -> None:
        """
        Lazily initialise log_lambda_var on first forward call.

        Initialised at log(0.5) so that Lambda = 0.5*I, which sets
        q(g_u) to the prior p(g_u) at the start of training (Section V.A).

        Parameters
        ----------
        n : int
            Number of training points.
        device : torch.device
        dtype : torch.dtype
        """
        self.log_lambda_var = torch.nn.Parameter(
            torch.full(
                (n,),
                fill_value=math.log(0.5),
                device=device,
                dtype=dtype,
            )
        )

    # -------------------------------------------------------------------------
    # Forward — computes q(g) at input x via Eq. 10-11
    # -------------------------------------------------------------------------

    def forward(self, x: Tensor) -> MultivariateNormal:
        """
        Compute the posterior q(g) at input locations x using the
        reparametrization of Liu et al. (2020), Eq. 10-11.

        This method:
            1. Computes mu_u, Sigma_u from log_lambda_var (Eq. 11)
            2. Propagates to q(g) at x via Eq. 10
            3. Caches mu_u, Sigma_u, sigma_g_diag for AddedLossTerms

        Only the diagonal of Sigma_g is computed to avoid O(n^2) memory.

        Parameters
        ----------
        x : Tensor, shape (n, d)
            Input locations (training points during training,
            test points during prediction).

        Returns
        -------
        MultivariateNormal
            q(g) at x, with mean mu_g (n,) and diagonal covariance
            DiagLinearOperator(sigma_g_diag) of shape (n, n).
        """
        n = x.size(0)

        # Lazy initialisation of log_lambda_var
        if self.log_lambda_var is None:
            self._init_lambda(n, device=x.device, dtype=x.dtype)

        # Inducing points and scalar prior mean
        X_u = self.variational_strategy.inducing_points          # (u, d)
        mu_0 = self.mean_module.constant                          # scalar

        # -----------------------------------------------------------------
        # Kernel matrices
        # -----------------------------------------------------------------
        K_uu = self.covar_module(X_u).to_dense()                 # (u, u)
        K_nu = self.covar_module(x, X_u).to_dense()              # (n, u)
        K_nn_diag = self.covar_module(x).diagonal()              # (n,)

        K_uu_inv = torch.linalg.inv(K_uu)                        # (u, u)
        Omega_nu = K_nu @ K_uu_inv                               # (n, u)

        # -----------------------------------------------------------------
        # Eq. 11 — reparametrized q(g_u)
        # -----------------------------------------------------------------
        lambda_diag = torch.exp(self.log_lambda_var)              # (n,), >= 0

        # Eq. 11a: mu_u = K_un (Lambda - 0.5*I) 1 + mu_0 * 1
        mu_u = K_nu.T @ (lambda_diag - 0.5) + mu_0               # (u,)

        # Eq. 11b: Sigma_u^{-1} = K_uu^{-1} + Omega_nu^T Lambda Omega_nu
        Sigma_u_inv = (
            K_uu_inv
            + Omega_nu.T @ torch.diag(lambda_diag) @ Omega_nu    # (u, u)
        )
        Sigma_u = torch.linalg.inv(Sigma_u_inv)                  # (u, u)

        # -----------------------------------------------------------------
        # Eq. 10 — posterior q(g) at x
        # -----------------------------------------------------------------

        # Eq. 10a: mu_g = Omega_nu (mu_u - mu_0*1) + mu_0*1
        mu_g = Omega_nu @ (mu_u - mu_0) + mu_0                   # (n,)

        # Eq. 10b: diag(Sigma_g) = diag(K_nn - Q_nn + Omega_nu Sigma_u Omega_nu^T)
        # Each term computed as a diagonal to avoid O(n^2) memory:
        #   diag(Q_nn)                 = row-wise dot product of Omega_nu and K_nu
        #   diag(Omega_nu Sigma_u Omega_nu^T) = row-wise dot of (Omega_nu @ Sigma_u)
        # and Omega_nu
        Q_nn_diag = (Omega_nu * K_nu).sum(dim=-1)                 # (n,)
        Sigma_g_diag = (
            K_nn_diag
            - Q_nn_diag
            + (Omega_nu @ Sigma_u * Omega_nu).sum(dim=-1)         # (n,)
        )

        # -----------------------------------------------------------------
        # Cache intermediates for AddedLossTerms
        # -----------------------------------------------------------------
        self._mu_u = mu_u
        self._Sigma_u = Sigma_u
        self._sigma_g_diag = Sigma_g_diag

        return MultivariateNormal(
            mu_g,
            DiagLinearOperator(Sigma_g_diag)
        )


# HETEOSKEDASTIC LIKELIHOOD MODEL
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
    def compute_rg(
        self,
        x_train: Tensor
    ) -> DiagLinearOperator:
        """
        Given the feature tensor, compute the added diagonal term, rg, \
            to be added to the covariance matrix during training.

        Parameters
        ----------
        x: Tensor
            The values of the training features to use for calling \
            the noise GP (g).

        Returns
        -------
            DiagLinearOperator
        """
        dist_g = self.noise_model(x_train)
        mu_g = dist_g.mean
        var_g = dist_g.variance  # Directly gives sigma**2

        return DiagLinearOperator(torch.exp(mu_g - .5 * var_g))

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
        # TODO: Modify function signature to be consistent with the
        # ELBO call.

        # Add the diagonal
        dist = MultivariateNormal(
            f_dist.mean,
            f_dist.lazy_covariance_matrix + self.compute_rg(x_train=x)
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


# MAIN GP
class VariationalSparseHeteroskedasticGP(GP, GPInterface):
    """
    Implemention of Spare Variational Heteroskedastic Gaussian Process \
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

    TODO: Finish description.

    Attributes
    ----------
    latent_f: SparseGP
        Instance of sparse GP for the latent function (f).
    latent_g: SparseGP
        Instance of sparse GP for the latent noise function (g).
    likelihood: HeteroskedasticLikelihood
        Instance of heteroskedastic likelihood.
    """

    def __init__(
        self,
        train_x: np.ndarray | Tensor,
        covar_f: Kernel | None = None,
        covar_g: Kernel | None = None,
        num_ind_f: int = 512,
        num_ind_g: int = 512,
        jitter: float = 1e-4,
    ):
        """
        Init class object.

        Parameters
        ----------
        train_x: np.ndarray | Tensor
            Input training features.
        covar_latent: Kernel | None
            (Optional) Kernel to be used for the covar module of latent GP (f). \
            Use the paper default if None is passed.
        covar_latent: Kernel | None
            (Optional) Kernel to be used for the covar module of latent GP (g). \
            Use the paper default if None is passed.
        num_ind_f: int
            Number of inducing points to be used for the latent GP (f).
        num_ind_g: int
            Number of inducing points to be used for the latent GP (g).
        jitter: float
            Small term to be added on the diagonal covariance for numerical \
            stability. Default is 1e-4.
        """
        super().__init__()
        # Set up the latent GP (f)
        train_x = [train_x, train_x[:, None]][train_x.ndim == 1]
        d = train_x.shape[-1]
        self.latent_f = SparseGP(
            get_inductions_points(train_x, num_ind_f),
            mean_module=ZeroMean(),
            covar_module=[covar_f, default_kernel(d)][covar_f is None]
        )

        # Set up the latent GP (g)
        self.latent_g = SparseGP(
            get_inductions_points(train_x, num_ind_g),
            mean_module=ConstantMean(),
            covar_module=[covar_g, default_kernel(d)][covar_g is None]
        )
        self.likelihood = HeteroskedasticLikelihood(self.latent_g)

        # Variational parameters
        n = to_tensor(train_x).size(0)
        self.log_lambda_var = torch.nn.Parameter(
            torch.ones(n) * np.log(.5)  # Force the mean of the var dir to be mu_g
        )

        # Misc
        self.jitter = jitter

    # Properties
    @property
    def lambda_var(self) -> torch.nn.Parameter:
        """Return the diagonal matrix Lambda."""
        return torch.diag(torch.exp(self.log_lambda_var))

    # Utils
    def compute_variational_params(
        self,
        train_x: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute variational parameters mu_u and Sigma_u from Lambda_nn \
            using the reparameterisation of Eq. 11 (Liu et al. 2020):

        >>> mu_u = K_un (Lambda - 0.5*I) 1 + mu_0 * 1
        >>> Sigma_u^{-1} = K_uu^{-1} + Omega_nu^T Lambda Omega_nu

        where Omega_nu = K_nu K_uu^{-1}

        Parameters
        ----------
        train_x : Tensor (n, d)
            Training features.

        Returns
        -------
        mu_u : Tensor (u,)
            Variational mean for q(g_u).
        Sigma_u : Tensor (u, u)
            Variational covariance for q(g_u).
        """
        # Evaluate the noise GP at the inducing points location
        x_u = self.latent_g.variational_strategy.inducing_points
        mu_g = self.latent_g.mean_module(x_u)
        k_uu = self.latent_g.covar_module(x_u).to_dense()

        # Compute the variational mean
        lmbd = self.lambda_var
        n = lmbd.size(0)
        eye = torch.eye(n, device=lmbd.device, dtype=lmbd.dtype)
        ones = torch.ones(n, device=mu_g.device, dtype=mu_g.dtype)
        k_un = self.latent_g.covar_module(x_u, train_x).to_dense()

        mu_u = k_un @ (self.lambda_var - 0.5 * eye) @ ones + mu_g

        # Compute the variational covar matrix
        kinv_uu = torch.inverse(_add_diag(k_uu, self.jitter))
        omega_nu = k_un.t() @ kinv_uu
        sigma_u = kinv_uu + omega_nu.t() @ self.lambda_var @ omega_nu

        return mu_u, sigma_u

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
        return self.latent_f.forward(X)

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
        raise NotImplementedError('Function not implemented yet.')

    def fit(
        self,
        train_x: np.ndarray | Tensor,
        train_y: np.ndarray | Tensor,
        n_epochs: int = 150,
        optim_kw: Mapping[str, Any] = {},
        pre_train: bool = True,
        pre_train_kw: Mapping[str, Any] = {},
        verbose: bool = True,
    ) -> 'VariationalSparseHeteroskedasticGP':
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
            # TODO: Add pre-training
            msg = 'Pre-training not implemented'
            logger.warning(msg)
            warnings.warn(msg)

        # Set training mode
        self.latent_f.train()
        self.latent_g.train()
        self.likelihood.train()

        # Setup optimizer
        optimizer = Adam(
            [
                {'params': self.latent_f.parameters()},
                {'params': self.latent_g.parameters()},
                {'params': [self.log_lambda_var]},
            ],
            **(_get_defaults() | optim_kw)
        )

        # Set the objective function
        # TODO: clarify is this is needed

        # Start training loop
        for n in range(n_epochs):
            # Zero grad
            optimizer.zero_grad()

            # Call
            q_f = self.latent_f(train_x)
            q_g = self.latent_g(train_y)

            # TODO: Comptute the log-probability term
            loss = self.likelihood.expected_log_prob(train_y, q_f, x=train_x)

            # Add the trace_g term
            loss -= .25 * q_g.variance.sum()

            # Add the 

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

    def predict_f(
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
        self.latent_f.eval()
        self.latent_g.eval()
        self.likelihood.eval()

        with torch.no_grad():
            f_dist = self.latent_f(test_x)
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
        self.latent_f.eval()
        self.latent_g.eval()
        self.likelihood.eval()

        with torch.no_grad():
            f_dist = self.latent_f(test_x)
            y_dist = self.likelihood(f_dist, x=test_x)
            lower, upper = y_dist.confidence_region()

        if return_ci:
            return y_dist, lower, upper

        return y_dist

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
