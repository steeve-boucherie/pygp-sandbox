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
import abc
import logging
import math
import warnings
from typing import Any, Callable, List, Literal, Mapping, Tuple

from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import Kernel, RBFKernel, ScaleKernel
from gpytorch.likelihoods import Likelihood
from gpytorch.means import ConstantMean, Mean, ZeroMean
from gpytorch.mlls._approximate_mll import _ApproximateMarginalLogLikelihood
from gpytorch.models import ApproximateGP
from gpytorch.optim import NGD
from gpytorch.utils.quadrature import GaussHermiteQuadrature1D
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    NaturalVariationalDistribution,
    # TrilNaturalVariationalDistribution,
    VariationalStrategy
)

from linear_operator.operators import DiagLinearOperator

import numpy as np

import pandas as pd

import torch
from torch import Tensor
from torch.distributions import Distribution, Normal
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset


# PACKAGE IMPORTS
# from gp_sand.means import MeanInterface
from gp_sand.distributions import GumbelDistribution
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


# 2D GHQ
class GaussHermiteQuadrature2D(GaussHermiteQuadrature1D):
    """
    An implementation of the Gauss-Hermite Quadrature to compute \
    2D dimensional integral, as requested by the ELBO of SVSHGP.

    Notes
    -----
    This class inheterits from the 1-D version in GPyTorch for simplicity.

    Attributes
    ----------
    num_locs: int
        Number of location points used for approximating the integral.
    """
    def __init__(self, num_locs: int = 20):
        super().__init__(num_locs)

    def forward(
        self,
        func: Callable[[Tensor], Tensor],
        dist_f: MultivariateNormal,
        dist_g: MultivariateNormal
    ) -> Tensor:
        """
        Given the distrubution samples, compute the 2D GHQ integral.

        Parameters
        ----------
        func: Callable[[Tensor, Tensor], Tensor]
            A callable computing the likelihood form the grid of distribution \
            parameters.
        dist_f: MultivariateNormal
            Samples from the latent function of the SVSGP.
        dist_g: MultivariateNormal
            Sample from the latent funnoisection of the SVSGP.
        """
        # Get the distribution moments
        mean_f, std_f = dist_f.mean, dist_f.stddev
        mean_g, std_g = dist_g.mean, dist_g.stddev
        nq = self.num_locs

        # Get the locs and weigths
        locs = self.locations.to(mean_f.dtype)
        weights = self.weights.to(mean_f.dtype)

        # Reshape to create the grid of distribution parameters.
        locs = locs.view(nq, *([1] * mean_f.dim()))
        f_nodes = mean_f.unsqueeze(0) + math.sqrt(2) * std_f.unsqueeze(0) * locs
        g_nodes = mean_g.unsqueeze(0) + math.sqrt(2) * std_g.unsqueeze(0) * locs

        f_grid = f_nodes.unsqueeze(1)   # (Nq, 1, *batch)
        g_grid = g_nodes.unsqueeze(0)   # (1, Nq, *batch)

        h_vals = func(f_grid, g_grid)   # (Nq, Nq, *batch)

        w_grid = (weights.view(nq, 1) * weights.view(1, nq)).view(nq, nq, *([1] * mean_f.dim()))
        return (w_grid * h_vals).sum(dim=(0, 1)) / math.pi


# LIKELIHOODS
class HeteroskedasticLikelihood(Likelihood, abc.ABC):
    """Abstract class for heteroskedastic likelihood models."""

    def __init__(self):
        super().__init__()

    # Methods
    @abc.abstractmethod
    def forward(self, f: Tensor, g: Tensor, *args: Any, **kwargs: Any) -> Distribution:
        """
        Return a torch.distributions object for p(y | f, g).

        Parameters
        ----------
        f: Tensor
            Tensor of estimated mean of the latent function.
        g: Tensor
            Tensor of estimated mean of the latent noise.
        """
        raise NotImplementedError('This is an abstract class!')

    @abc.abstractmethod
    def expected_log_prob(
        self,
        observations: Tensor,
        dist_f: MultivariateNormal,
        dist_g: MultivariateNormal,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        """
        Given the approximate distributions, compute the expected log-probability\
        of the observed data.

        Parameters
        ----------
        target: Tensor
            Values of observed data..
        dist_f: MultivariateNormal
            Estimate latent function.
        dist_g: MultivariateNormal
            Estimate latent noise.

        Returns
        -------
            Tensor
        """
        raise NotImplementedError('This is an abstract class!')

    @abc.abstractmethod
    def marginal(
        self,
        dist_f: MultivariateNormal,
        dist_g: MultivariateNormal,
        *args: Any,
        **kwargs: Any
    ) -> Distribution:
        """
        Given the approximate distributions, compute marginal \
        distribution by adding the noise.

        Parameters
        ----------
        dist_f: MultivariateNormal
            Estimate latent function.
        dist_g: MultivariateNormal
            Estimate latent noise.

        Returns
        -------
            Tensor
        """
        raise NotImplementedError('This is an abstract class!')


class _TwoDimensionalLikelihood(HeteroskedasticLikelihood):
    """
    Abstract class for likelihoods p(y | f, g) driven by two conditionally-
    independent latent GPs (e.g. f -> location, g -> scale), leverage \
    2-dimensional Gauss-Hermite Quadrature to approximate the integral.

    Parameters
    ----------
    num_locs: int
        Number of location points used for approximating the integral. \
        Default is 20.
    """

    def __init__(
        self,
        num_locs: int = 20,
    ):
        super().__init__()
        self.quadrature = GaussHermiteQuadrature2D(num_locs)

    # Methods
    def expected_log_prob(
        self,
        observations: Tensor,
        dist_f: MultivariateNormal,
        dist_g: MultivariateNormal,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        """
        Given the approximate distributions, compute the expected log-probability\
        of the observed data.

        Parameters
        ----------
        target: Tensor
            Values of observed data..
        dist_f: MultivariateNormal
            Estimate latent function.
        dist_g: MultivariateNormal
            Estimate latent noise.

        Returns
        -------
            Tensor
        """
        def log_prob_func(f, g):
            dist: Distribution = self.forward(f, g, *args, **kwargs)
            return dist.log_prob(observations)
        return self.quadrature(log_prob_func, dist_f, dist_g)


class HeteroskedasticGaussianLikelihood(HeteroskedasticLikelihood):
    """
    Implementation of heteroskedastic likelihoodfor normally-distributed \
    observation noise that can be written in closed form.
    """

    def __init__(self):
        super().__init__()

    # Methods
    def forward(self, f: Tensor, g: Tensor, *args: Any, **kwargs: Any) -> GumbelDistribution:
        raise NotImplementedError('This is not needed for the Heteroskedastic Gaussian')

    def expected_log_prob(
        self,
        observations: Tensor,
        dist_f: MultivariateNormal,
        dist_g: MultivariateNormal,
        *args: Any,
        **kwargs: Any,
    ) -> Tensor:
        """
        Given the approximate distributions, compute the expected log-probability\
        of the observed data.

        Parameters
        ----------
        target: Tensor
            Values of observed data..
        dist_f: MultivariateNormal
            Estimate latent function.
        dist_g: MultivariateNormal
            Estimate latent noise.

        Returns
        -------
            Tensor
        """
        # NOTE: The sum is handled by the ELBO class
        # Get the noise-GP predictions
        rg = torch.exp(dist_g.mean - 0.5 * dist_g.variance)

        # Term : main likelihood (residuals)
        llk = Normal(dist_f.mean, rg.sqrt()).log_prob(observations)  # .sum()

        # Term 2: trace of G
        trace_g = 0.25 * dist_g.variance  # .sum()

        # Term 3: trace of F scaled with the Rg posterior
        trace_f = 0.5 * (dist_f.variance / rg)  # .sum()

        return llk - trace_g - trace_f

    def marginal(
        self,
        dist_f: MultivariateNormal,
        dist_g: MultivariateNormal,
        *args: Any,
        **kwargs: Any,
    ) -> GumbelDistribution:
        mean_f, covar_f = dist_f.mean, dist_f.lazy_covariance_matrix
        mean_g, var_g = dist_g.mean, dist_g.variance
        noise_covar = DiagLinearOperator(torch.exp(mean_g + .5 * var_g))

        return MultivariateNormal(mean_f, covar_f + noise_covar)


class HeteroskedasticGumbelLikelihood(_TwoDimensionalLikelihood):
    """
    Implementation of heteroskedastic likelihood for Gumbel-distributed \
    observation noise.

    Parameters
    ----------
    num_locs: int
        Number of location points used for approximating the integral. \
        Default is 20.
    beta_link: Callable[[Tensor], Tensor]
        A function to transform the unconstrained GP estimates into to \
        the value space of the scale parameter, beta, that must be positive.
        Default is the softplus method.
    """
    def __init__(
        self,
        beta_link: Callable[[Tensor], Tensor] = torch.nn.functional.softplus,
        # beta_link: Callable[[Tensor], Tensor] = torch.exp,
        num_locs: int = 20
    ):
        super().__init__(num_locs=num_locs)
        self.beta_link = beta_link

    def forward(self, f: Tensor, g: Tensor, *args: Any, **kwargs: Any) -> GumbelDistribution:
        return GumbelDistribution(loc=f, scale=self.beta_link(g), **kwargs)

    def marginal(
        self,
        dist_f: MultivariateNormal,
        dist_g: MultivariateNormal,
        *args: Any,
        **kwargs: Any,
    ) -> GumbelDistribution:
        loc = dist_f.mean
        scale = self.beta_link(dist_g.mean)
        return GumbelDistribution(loc, scale)


# NOISE GP
class NoiseGP(ApproximateGP):
    """
    Variational Sparse Gaussian Process for modeling heteroskedastic observation noise.

    This class implements the log-noise GP g(x) in the heteroskedastic GP framework, \
    where the input-dependent noise variance is modeled as σ²(x) = exp(g(x)). The \
    sparse approximation uses inducing points and variational inference to enable \
    scalable training on large datasets.

    Model Definition
    ----------------
    The log-noise function follows a Gaussian process prior:

        g(x) ~ GP(μ₀, k_g(x, x'))

    where:
        μ₀      : Prior mean (learnable constant representing average log-noise level)
        k_g     : Kernel function (e.g., RBF with ARD for capturing noise patterns)
        σ²(x)   : exp(g(x)) is the heteroskedastic noise variance

    Unlike the latent function f(x) which uses a zero-mean prior, g(x) requires a \
    non-zero prior mean μ₀ to account for the baseline noise level, since we cannot \
    assume the noise variance is centered at 1 (i.e., g(x) centered at 0).

    Sparse Approximation
    --------------------
    To scale to large datasets, the full GP is approximated using u inducing points \
    {X_u, g_u}:

        p(g|g_u) = N(g | Ω_u(g_u - μ₀1) + μ₀1, K_gg - Q_gg)

    where:
        Ω_u = K_gu K_uu⁻¹           : Projection matrix
        Q_gg = K_gu K_uu⁻¹ K_ug     : Low-rank approximation to K_gg
        K_gg - Q_gg                 : Approximation error (trace penalty)

    The inducing variables g_u are approximated by a variational distribution:

        q(g_u) = N(g_u | μ_u, Σ_u)

    with variational parameters {μ_u, Σ_u} and inducing locations X_u learned \
    jointly during training.

    Predictive Distribution
    -----------------------
    For a test point x*, the approximate posterior is:

        q(g*) = ∫ p(g*|g_u) q(g_u) dg_u = N(g* | μ_g*, σ²_g*)

    with:
        μ_g* = k*u K_uu⁻¹ (μ_u - μ₀1) + μ₀
        σ²_g* = k** - k*u K_uu⁻¹ k_u* + k*u K_uu⁻¹ Σ_u K_uu⁻¹ k_u*

    The predicted noise variance at x* is then:

        σ²(x*) = exp(μ_g* + 0.5 * σ²_g*)

    This accounts for both the mean prediction and uncertainty in g.

    Parameters
    ----------
    ind_points : Tensor, shape (u, d)
        Inducing point locations for the sparse approximation. These are variational \
        parameters that can be optimized during training when learn_inducing_locations=True.

    mean_module : Mean
        Prior mean function for g(x). Typically a ConstantMean() representing μ₀, \
        which captures the average log-noise level across the input space.

    covar_module : Kernel
        Kernel (covariance) function for g(x). Defines the smoothness and structure \
        of noise variations. Common choice: ScaleKernel(RBFKernel(ard_num_dims=d)).

    use_ngd: bool, default=False
        An option for whether to use the natural gradient descent (NGD) during training \
        of the variational distribution parameters. This can speed-up convergence \
        significantly but cause numerical instabilities. Use with caution.

    jitter_val : float or None, default=None
        Jitter value added to the diagonal of kernel matrices for numerical stability. \
        Helps prevent issues with near-singular matrices during Cholesky decomposition. \
        If None, uses GPyTorch's default jitter (typically 1e-6).

    Attributes
    ----------
    mean_module : Mean
        The prior mean function μ₀.

    covar_module : Kernel
        The kernel function k_g for modeling noise correlations.

    variational_strategy : VariationalStrategy
        Contains the inducing points X_u and variational distribution q(g_u). \
        Handles the sparse GP approximation and KL divergence computation.

    Methods
    -------
    forward(x)
        Compute the approximate posterior q(g|x) for the log-noise at input x. \
        Returns a MultivariateNormal distribution.

    added_noise(x)
        Compute the added noise covariance to be added to the latent function's \
        posterior. Returns a diagonal operator with entries exp(μ_g + 0.5 * σ²_g).

    Usage in SVSHGP
    ---------------
    The NoiseGP is used within SVSHGP to:

    1. **During training**: Provide the noise variance R_g for computing the likelihood
       term in the ELBO:

           R_g = exp(μ_g - 0.5 * σ²_g)  # Effective noise for likelihood

    2. **During prediction**: Add heteroskedastic uncertainty to the latent function's
       prediction:

           Var[y*] = Var[f*] + exp(μ_g* + 0.5 * σ²_g*)

    The difference in the exponential terms arises from different contexts:
    - Training uses R_g for the likelihood mean (Jensen's inequality correction)
    - Prediction uses full noise variance including g's uncertainty

    Examples
    --------
    >>> import torch
    >>> from gpytorch.means import ConstantMean
    >>> from gpytorch.kernels import ScaleKernel, RBFKernel

    >>> # Create inducing points (e.g., via k-means or random subset)
    >>> u, d = 30, 5
    >>> inducing_points = torch.randn(u, d)

    >>> # Initialize noise GP
    >>> noise_gp = NoiseGP(
    ...     ind_points=inducing_points,
    ...     mean_module=ConstantMean(),
    ...     covar_module=ScaleKernel(RBFKernel(ard_num_dims=d))
    ... )

    >>> # Forward pass: get posterior distribution at training points
    >>> x_train = torch.randn(100, d)
    >>> g_dist = noise_gp(x_train)
    >>> print(f"Log-noise mean: {g_dist.mean.shape}")
    >>> print(f"Log-noise variance: {g_dist.variance.shape}")

    >>> # Get noise variance for prediction
    >>> noise_gp.eval()
    >>> with torch.no_grad():
    ...     x_test = torch.randn(20, d)
    ...     added_noise_cov = noise_gp.added_noise(x_test)
    ...     noise_variance = added_noise_cov.diagonal()
    >>> print(f"Predicted noise variance: {noise_variance[:5]}")

    Notes
    -----
    - **Mean function**: Unlike f(x), the noise GP requires a non-zero mean μ₀ since \
      noise variance σ² = exp(g) should not be assumed to equal 1 everywhere.

    - **Inducing points**: The number of inducing points u controls the approximation \
      quality. Fewer points than for f(x) may suffice since noise often varies more \
      smoothly than the latent function. Typical: u ∈ [0.005n, 0.05n].

    - **Kernel choice**: The kernel for g can differ from f's kernel. Noise patterns \
      may have different length scales or smoothness properties than the latent function.

    - **Independence assumption**: NoiseGP is trained independently from the latent GP \
      (separate variational distribution q(g_u)), though both contribute to the joint ELBO.

    - **Positivity constraint**: Using g = log(σ²) ensures positivity of noise variance \
      without constrained optimization. The exponential transformation is applied in \
      the likelihood computation and prediction.

    - **Training vs. prediction**: Note the different exponential terms:
      * Training: R_g = exp(μ_g - 0.5σ²_g) for stable likelihood evaluation
      * Prediction: σ² = exp(μ_g + 0.5σ²_g) for full predictive uncertainty

    - **Initialization**: The constant mean μ₀ should be initialized near log(σ²_empirical)
      where σ²_empirical is the empirical variance of residuals from a preliminary fit.

    Computational Complexity
    ------------------------
    - Forward pass: O(|B|u²) for a mini-batch of size |B|
    - KL divergence: O(u³) computed once per iteration
    - Total per iteration: O(|B|u² + u³)

    This is much more efficient than exact GP's O(n³) complexity.

    Mathematical Details
    --------------------
    The variational posterior q(g) marginalizes over the inducing variables:

        q(g) = ∫ p(g|g_u) q(g_u) dg_u

    This integral is analytically tractable for Gaussian distributions:

        q(g) = N(g | μ_g, Σ_g)

    where:
        μ_g = Ω_u(μ_u - μ₀1) + μ₀1
        Σ_g = K_gg - Q_gg + Ω_u Σ_u Ω_u^T

    The KL divergence KL[q(g_u)||p(g_u)] provides regularization to prevent \
    overfitting and is computed analytically using the Gaussian KL formula.

    References
    ----------
    Liu, H., Ong, Y. S., & Cai, J. (2020). Large-scale Heteroscedastic Regression \
    via Gaussian Process. arXiv preprint arXiv:1811.01179v3.

    Titsias, M. (2009). Variational learning of inducing variables in sparse \
    Gaussian processes. In AISTATS.

    See Also
    --------
    SVSHGP : Main heteroskedastic GP model that uses NoiseGP
    SVSHGPVariationalELBO : ELBO implementation for joint training
    gpytorch.models.ApproximateGP : Base class for variational sparse GPs
    """

    def __init__(
        self,
        ind_points: Tensor,
        mean_module: Mean,
        covar_module: Kernel,
        use_ngd: bool = False,
        jitter_val: float | None = None,
    ):
        """Init class."""
        var_dist = [
            CholeskyVariationalDistribution,
            NaturalVariationalDistribution,
        ][use_ngd]
        var_strat = VariationalStrategy(
            self,
            inducing_points=ind_points,
            variational_distribution=var_dist(
                num_inducing_points=ind_points.size(0)
            ),
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
        # DEPRECATE: Remove this class
        msg = 'This method is deprecated and will be remove. Use the specific ' \
              'class "HeteroskedasticGaussianLikelihood" to handle heteroskedastic' \
              'noise.'
        logger.warning(msg)
        warnings.warn(msg, DeprecationWarning, 2)

        dist_g = self(x)
        mu_g = dist_g.mean
        var_g = dist_g.variance  # Directly gives sigma**2

        return DiagLinearOperator(torch.exp(mu_g + .5 * var_g))


# MLL
class SVSHGPVariationalELBO(_ApproximateMarginalLogLikelihood):
    """
    Variational Evidence Lower Bound (ELBO) for Stochastic Variational Sparse \
        Heteroskedastic Gaussian Process.

    This class implements the factorized ELBO objective from Liu et al. (2020) \
    that enables efficient stochastic variational inference for heteroskedastic \
    GP regression with mini-batching support.

    Objective Function
    ------------------
    The ELBO is factorized over training data points to enable mini-batch optimization:

        F = Σᵢ₌₁ⁿ E_{q(fᵢ)q(gᵢ)}[log p(yᵢ|fᵢ, gᵢ)] \
            - KL[q(f_m) || p(f_m)] \
            - KL[q(g_u) || p(g_u)]

    For a mini-batch B ⊂ {1, ..., n}, the unbiased stochastic estimate is:

        F̃ = (n/|B|) Σᵢ∈B E_{q(fᵢ)q(gᵢ)}[log p(yᵢ|fᵢ, gᵢ)] \
            - KL[q(f_m) || p(f_m)] \
            - KL[q(g_u) || p(g_u)]

    where |B| is the mini-batch size and n is the total training size.

    Likelihood Term Decomposition
    ------------------------------
    The expected log-likelihood for each data point decomposes as:

        E_{q(fᵢ)q(gᵢ)}[log p(yᵢ|fᵢ, gᵢ)] = L₁ᵢ - L₂ᵢ - L₃ᵢ

    where:
        L₁ᵢ = log N(yᵢ | μ_fᵢ, R_gᵢ)           # Main likelihood term
        L₂ᵢ = 0.25 * σ²_gᵢ                    # Variance correction for g
        L₃ᵢ = 0.5 * σ²_fᵢ / R_gᵢ              # Variance correction for f

    with:
        μ_fᵢ, σ²_fᵢ  : mean and variance of q(fᵢ) from the latent GP
        μ_gᵢ, σ²_gᵢ  : mean and variance of q(gᵢ) from the noise GP
        R_gᵢ = exp(μ_gᵢ - 0.5 * σ²_gᵢ)  : effective noise variance

    The variance correction terms (L₂, L₃) arise from the variational \
        approximation and ensure the ELBO is a valid lower bound.

    KL Divergence Terms
    -------------------
    Two KL divergence terms regularize the variational distributions:

        KL[q(f_m) || p(f_m)] : Regularizes the latent function inducing variables
        KL[q(g_u) || p(g_u)] : Regularizes the noise function inducing variables

    These terms prevent the variational distributions from deviating too far from \
        their priors and are computed analytically for Gaussian distributions.

    Normalization Convention (GPyTorch)
    -----------------------------------
    Following GPyTorch conventions, the implementation normalizes all terms:

        F̃_norm = (1/|B|) Σᵢ∈B E[log p(yᵢ|fᵢ, gᵢ)] \
                 - (1/n) KL[q(f_m) || p(f_m)] \
                 - (1/n) KL[q(g_u) || p(g_u)]

    This normalization:
    - Improves numerical stability for large datasets
    - Makes learning rates more transferable across dataset sizes
    - Maintains gradient directions (since F̃_norm = F̃/n)

    Parameters
    ----------
    likelihood: _TwoDimensionalLikelihood
        The likelihood model for the data. It must be a subclass of `_TwoDimensionalLikelihood`
        and implement the `forward` method called in `expected_log_prob`.
    model : SVSHGP
        The SVSHGP model instance containing both the latent GP (f) and \
        noise GP (g) with their variational strategies.
    num_data : int
        Total number of training data points (n). Used to properly scale \
        the KL divergence terms relative to the mini-batch likelihood.
    beta : float, default=1.0
        KL divergence weighting factor. Values < 1 reduce KL penalty \
        (increasing model flexibility), values > 1 increase regularization.
        Typically kept at 1.0 for standard variational inference.
    combine_terms : bool, default=True
        If True, returns the complete ELBO as a single scalar.
        If False, returns individual components (log_likelihood, kl_f, kl_g, log_prior) \
        for debugging and analysis.

    Attributes
    ----------
    num_data : int
        Total training dataset size.
    beta : float
        KL divergence scaling factor.
    combine_terms : bool
        Whether to combine ELBO terms into a single value.

    Methods
    -------
    _log_likelihood_term(dist_f, targets, **kwargs)
        Computes the expected log-likelihood term with variance corrections. \
        Requires kwargs['x'] containing the input features for evaluating g(x).

    forward(dist_f, targets, **kwargs)
        Computes the complete ELBO (or its components if combine_terms=False). \
        This is the main objective function maximized during training.

    Mathematical Derivation
    -----------------------
    The likelihood term derivation starts from:

        p(y|f, g) = N(y | f, exp(g))

    Taking expectations under q(f)q(g):

        E[log p(y|f, g)] = E[log N(y | f, exp(g))]
                         = E[-0.5 * log(2π) - 0.5 * g - 0.5 * (y-f)²/exp(g)]

    Applying the variational distributions q(f) = N(μ_f, σ²_f) and \
    q(g) = N(μ_g, σ²_g), and using:

        E[(y-f)²] = (y - μ_f)² + σ²_f
        E[exp(-g)] = exp(-μ_g + 0.5 * σ²_g)

    yields the three-term decomposition implemented in _log_likelihood_term.

    Examples
    --------
    >>> import torch
    >>> from gpytorch.distributions import MultivariateNormal

    >>> # Assume model is a trained SVSHGP instance
    >>> mll = SVSHGPVariationalELBO(model, num_data=1000, beta=1.0)

    >>> # During training with mini-batch
    >>> batch_x = torch.randn(64, 5)  # Mini-batch of 64 samples
    >>> batch_y = torch.randn(64)

    >>> # Forward pass through model
    >>> model.train()
    >>> output_f = model(batch_x)

    >>> # Compute ELBO (to be maximized, so we minimize its negative)
    >>> loss = -mll(output_f, batch_y, x=batch_x)

    >>> # For debugging, get individual components
    >>> mll.combine_terms = False
    >>> ll, kl_f, kl_g, prior = mll(output_f, batch_y, x=batch_x)
    >>> print(f"Log-likelihood: {ll.item():.4f}")
    >>> print(f"KL(q(f_m)||p(f_m)): {kl_f.item():.4f}")
    >>> print(f"KL(q(g_u)||p(g_u)): {kl_g.item():.4f}")

    Notes
    -----
    - The ELBO must be **maximized**, so use `loss = -mll(...)` for gradient descent.
    - The `x=batch_x` keyword argument is **required** in the forward call to \
      evaluate the noise GP g(x) for the current mini-batch.
    - KL divergence terms are the same for all mini-batches (they don't depend on data), \
      so they're computed once per iteration regardless of mini-batch size.
    - For very large datasets, the KL terms become negligible relative to the \
      likelihood, so the model is primarily data-driven.
    - The variance correction terms (L₂, L₃) are crucial for capturing \
      heteroskedasticity; removing them would degrade to homoskedastic GP.

    Computational Complexity
    ------------------------
    Per mini-batch iteration:
    - Likelihood term: O(|B|m² + |B|u²) for computing q(f) and q(g) on batch
    - KL divergence: O(m³ + u³) for the two variational distributions (computed once)
    - Total: O(|B|m² + |B|u² + m³ + u³)

    This is much more efficient than the exact GP complexity O(n³) or the \
    deterministic sparse variant O(nm² + nu²) when |B| << n.

    References
    ----------
    Liu, H., Ong, Y. S., & Cai, J. (2020). Large-scale Heteroscedastic Regression \
    via Gaussian Process. arXiv preprint arXiv:1811.01179v3.

    Hensman, J., Fusi, N., & Lawrence, N. D. (2013). Gaussian processes for big data.
    In Uncertainty in Artificial Intelligence (UAI).

    See Also
    --------
    SVSHGP : The main model class that uses this ELBO
    NoiseGP : Sparse GP for the noise variance function g(x)
    gpytorch.mlls.VariationalELBO : Standard homoskedastic variational ELBO
    """

    def __init__(
        self,
        likelihood: HeteroskedasticLikelihood,
        model: 'SVSHGP',
        num_data: int,
        beta: float = 1.,
        combine_terms: bool = True
    ):
        # NOTE: Use a dummy gaussian likelihood for super().__init__
        # This is ignored as the LL term is computed manually.
        super().__init__(likelihood, model, num_data, beta, combine_terms)

    def _log_likelihood_term(
        self,
        dist_f: MultivariateNormal,
        dist_g: MultivariateNormal,
        targets: Tensor,
        **kwargs
    ) -> Tensor:
        """
        Compute the log likelihood term of the ELBO.

        Attributes
        ----------
        dist_f: MultivariateNormal
            Estimated latent function.
        dist_g: MultivariateNormal
            Estimated latent noise.
        target: Tensor
            Training target values.
        kwargs: Mapping[str, Any]
            Optional argument it must constain the tensor of training features \
            to estimate the diagonal term.
        # """
        llk = self.likelihood.expected_log_prob(
            targets,
            dist_f,
            dist_g,
            **kwargs
        )
        return llk.sum(-1)

    def forward(
        self,
        dist_f: MultivariateNormal,
        dist_g: MultivariateNormal,
        targets: Tensor,
        **kwargs: Mapping[str, Any]
    ) -> Tensor:
        """
        Given the approximate distributions and training targets, compute the variational ELBO\
            including the KL divergence terms for both the latent and the noise GP.

        Notes
        -----
        This is a copy/past of the `foward` method of the `_ApproximateMarginalLogLikelihood` \
            whit the addition of the KL divergence term for the noise GP variational \
            distribution.

        Parameters
        ----------
        dist_f: MultivariateNormal
            Estimate latent function.
        dist_g: MultivariateNormal
            Estimate latent noise.
        target: Tensor
            Training targets.
        **kwargs: Mapping[str, Any]
            Optional argument to pass to the _log_likelihood_term function

        Returns
        -------
            Tensor
        """
        # Get likelihood term and KL term
        num_batch = dist_f.event_shape[0]
        log_likelihood = (
            self._log_likelihood_term(
                dist_f,
                dist_g,
                targets,
                **kwargs
            )
            .div(num_batch)
        )
        kl_divergence = (
            self.model
            .variational_strategy
            .kl_divergence()
            .div(self.num_data / self.beta)
        )

        # NOTE: This is the addition of the KL term of the
        # variational distrib of noise GP
        kl_div_g = (
            self.model
            .noise_gp
            .variational_strategy
            .kl_divergence()
            .div(self.num_data / self.beta)
        )

        # Add any additional registered loss terms
        added_loss = torch.zeros_like(log_likelihood)
        had_added_losses = False
        for added_loss_term in self.model.added_loss_terms():
            added_loss.add_(added_loss_term.loss())
            had_added_losses = True

        # Log prior term
        log_prior = torch.zeros_like(log_likelihood)
        for name, module, prior, closure, _ in self.named_priors():
            log_prior.add_(prior.log_prob(closure(module)).sum().div(self.num_data))

        if self.combine_terms:
            return log_likelihood - kl_divergence - kl_div_g + log_prior - added_loss
        else:
            if had_added_losses:
                return log_likelihood, kl_divergence, kl_div_g, log_prior, added_loss
            else:
                return log_likelihood, kl_divergence, kl_div_g, log_prior


# STOCHASTIC VARIATIONAL SPARSE HETEROSKEDASTIC GAUSSIAN PROCESS
class SVSHGP(ApproximateGP, GPInterface):
    """
    Stochastic Variational Sparse Heteroskedastic Gaussian Process (SVSHGP).

    This model implements the scalable heteroskedastic GP regression framework \
    from Liu et al. (2020), which handles input-dependent noise variance through \
    two independent sparse Gaussian processes with stochastic variational inference.

    Model Structure
    ---------------
    The heteroskedastic regression model is defined as:

        y(x) = f(x) + ε(x),  where ε(x) ~ N(0, σ²(x))

    where both the latent function and noise variance are modeled as GPs:

        f(x) ~ GP(0, k_f(x, x'))           # Latent function (zero mean)
        g(x) ~ GP(μ₀, k_g(x, x'))          # Log-noise process
        σ²(x) = exp(g(x))                  # Heteroskedastic noise variance

    Sparse Approximation
    --------------------
    To enable scalability, the model uses sparse approximations for both GPs:

    - **f(x)**: Approximated using m inducing points {X_m, f_m} with \
      variational distribution q(f_m) = N(f_m | μ_m, Σ_m)

    - **g(x)**: Approximated using u inducing points {X_u, g_u} with \
      variational distribution q(g_u) = N(g_u | μ_u, Σ_u)

    The inducing points and variational parameters are learned jointly during training.

    Variational Inference
    ---------------------
    The model maximizes the Evidence Lower Bound (ELBO):

        F = Σᵢ E_{q(fᵢ)q(gᵢ)}[log p(yᵢ|fᵢ, gᵢ)] \
            - KL[q(f_m) || p(f_m)] \
            - KL[q(g_u) || p(g_u)]

    where the likelihood term decomposes as:

        E[log p(yᵢ|fᵢ, gᵢ)] = log N(yᵢ | μ_fᵢ, R_gᵢ) \
                              - 0.25 * σ²_gᵢ \
                              - 0.5 * σ²_fᵢ / R_gᵢ

    with:
        μ_f, σ²_f : mean and variance of q(f)
        μ_g, σ²_g : mean and variance of q(g)
        R_g = exp(μ_g - 0.5 * σ²_g)

    Stochastic Optimization
    -----------------------
    The factorized ELBO enables efficient stochastic variational inference using \
    mini-batches, reducing time complexity from O(nm² + nu²) to O(|B|m² + |B|u² + m³ + u³) \
    where |B| << n is the mini-batch size.

    Parameters
    ----------
    ind_points_f : np.ndarray or Tensor, shape (m, d)
        Inducing points for the latent function f. These are variational parameters \
        that will be optimized during training (when learn_inducing_locations=True).

    ind_points_g : np.ndarray or Tensor, shape (u, d)
        Inducing points for the log-noise function g. Independent from ind_points_f \
        to allow different resolutions for modeling mean and variance.

    mean_f : Mean, default=ZeroMean()
        Mean function for the latent GP f(x). Standard choice is zero-mean after \
        data normalization.

    covar_f : Kernel or None, default=None
        Kernel function for the latent GP f(x). If None, uses ScaleKernel(RBFKernel) \
        with ARD (automatic relevance determination).

    mean_g : Mean, default=ConstantMean()
        Mean function for the log-noise GP g(x). Uses a learnable constant μ₀ to \
        account for the average noise level.

    covar_g : Kernel or None, default=None
        Kernel function for the log-noise GP g(x). If None, uses ScaleKernel(RBFKernel) \
        with ARD. Can differ from covar_f to capture different length scales.

    use_ngd: bool, default=False
        An option for whether to use the natural gradient descent (NGD) during training \
        of the variational distribution parameters. This can speed-up convergence \
        significantly but cause numerical instabilities. Use with caution.

    jitter_val : float or None, default=None
        Jitter value added to diagonal of covariance matrices for numerical stability. \
        If None, uses GPyTorch's default jitter.

    Attributes
    ----------
    mean_module : Mean
        Mean function for f(x).

    covar_module : Kernel
        Kernel function for f(x).

    noise_gp : NoiseGP
        Independent sparse GP for modeling log-noise variance g(x).

    variational_strategy : VariationalStrategy
        Variational strategy for the latent function f, containing inducing points
        and variational distribution q(f_m).

    Methods
    -------
    forward(x)
        Compute the approximate posterior distribution q(f|x) for the latent function.

    fit(train_x, train_y, n_epochs, batch_size, ...)
        Train the model using stochastic variational inference with mini-batching.

    predict(test_x, return_ci=True)
        Make predictions at test points, including heteroskedastic noise uncertainty.

    predict_latent(test_x, return_ci=True)
        Predict only the latent function f(x) without noise.

    score(test_x, test_y, methods)
        Evaluate model performance on test data.

    Examples
    --------
    >>> import torch
    >>> import numpy as np
    >>> from gpytorch.kernels import ScaleKernel, RBFKernel

    >>> # Generate synthetic heteroskedastic data
    >>> n, d = 1000, 5
    >>> X = np.random.randn(n, d)
    >>> y = np.sin(X[:, 0]) + np.random.randn(n) * np.exp(0.5 * X[:, 1])

    >>> # Initialize inducing points (e.g., via k-means or random subset)
    >>> m, u = 50, 30
    >>> ind_f = X[np.random.choice(n, m, replace=False)]
    >>> ind_g = X[np.random.choice(n, u, replace=False)]

    >>> # Create model
    >>> model = SVSHGP(
    ...     ind_points_f=ind_f,
    ...     ind_points_g=ind_g,
    ...     covar_f=ScaleKernel(RBFKernel(ard_num_dims=d)),
    ...     covar_g=ScaleKernel(RBFKernel(ard_num_dims=d))
    ... )

    >>> # Train with mini-batching
    >>> model.fit(X, y, n_epochs=100, batch_size=128, verbose=True)

    >>> # Make predictions
    >>> X_test = np.random.randn(100, d)
    >>> pred_dist, lower, upper = model.predict(X_test)
    >>> print(f"Predicted mean shape: {pred_dist.mean.shape}")
    >>> print(f"Predicted variance (heteroskedastic): {pred_dist.variance[:5]}")

    Notes
    -----
    - The model requires normalized data (zero mean, unit variance) for best performance.
    - Inducing point initialization affects convergence; consider k-means clustering or
      greedy variance reduction methods.
    - The number of inducing points (m, u) controls the trade-off between accuracy and
      computational cost. Typical choices: m, u ∈ [0.01n, 0.1n].
    - Mini-batch size should be large enough to provide stable gradient estimates,
      typically |B| ∈ [64, 512].
    - Use different inducing point sets for f and g to allow independent resolution
      control for mean and variance modeling.

    References
    ----------
    Liu, H., Ong, Y. S., & Cai, J. (2020). Large-scale Heteroscedastic Regression \
    via Gaussian Process. arXiv preprint arXiv:1811.01179v3.

    See Also
    --------
    NoiseGP : Sparse GP for modeling log-noise variance g(x)
    SVSHGPVariationalELBO : Custom ELBO implementation for joint training
    """

    def __init__(
        self,
        ind_points_f: np.ndarray | Tensor,
        ind_points_g: np.ndarray | Tensor,
        mean_f: Mean = ZeroMean(),
        covar_f: Kernel | None = None,
        mean_g: Mean = ConstantMean(),
        covar_g: Kernel | None = None,
        likelihood: HeteroskedasticLikelihood = HeteroskedasticGaussianLikelihood(),
        use_ngd: bool = False,
        jitter_val: float | None = None,
    ):
        """Init class."""
        # Set up the latent f GP
        var_dist = [
            CholeskyVariationalDistribution,
            NaturalVariationalDistribution,
        ][use_ngd]
        self.use_ngd = use_ngd  # Store for setup during training.
        var_strat = VariationalStrategy(
            self,
            inducing_points=ind_points_f,
            variational_distribution=var_dist(
                num_inducing_points=ind_points_f.size(0)
            ),
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
            use_ngd=use_ngd,
            jitter_val=jitter_val
        )

        self.likelihood = likelihood

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
        n_epochs: int = 5,
        batch_size: int = 64,
        adam_lr: float = 0.01,
        ngd_lr: float = 0.1,
        batch_kw: Mapping[str, Any] = {},
        optim_kw: Mapping[str, Any] = {},
        verbose: bool = True,
    ) -> 'SVSHGP':
        """
        Given the traning data and fitting option, fit the model.

        Parameters
        ----------
        train_x: np.ndarray | torch.Tensor, shape (n, m)
            Tensor of training features.
        train_y: np.ndarray | torch.Tensor, shape (n, m)
            Tensor of training targets.
        n_epochs: int
            Number of training epochs.
        batch_size: int
            Batch size to split the training data in mini-batches.
        adam_lr: float
            Learning rate for the Adam optimizer.
        ngd_lr: float
            Learning rate for the Natural Gradient Descent (NGD) optimizer. \
            This is only used if self.use_ngd is set to True.
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
            SVSHGP
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
                'optim': {'lr': adam_lr}
            }[obj]
            return params

        # Force input types
        train_x, train_y = [to_tensor(t) for t in [train_x, train_y]]

        # Set training mode
        self.train()
        self.noise_gp.train()

        # Set up the optimizer(s)
        if self.use_ngd:
            optimizers = [
                # Optimizer for variational parameters
                NGD(
                    self.variational_parameters(),
                    num_data=train_y.size(0),
                    lr=ngd_lr
                ),
                # Optimizer for the remainder
                Adam(
                    self.hyperparameters(),
                    **(_get_defaults('optim') | optim_kw)
                )
            ]

        else:
            optimizers = [Adam(
                self.parameters(),
                **(_get_defaults('optim') | optim_kw)
            )]

        # Set the objective function
        mll = SVSHGPVariationalELBO(self.likelihood, self, num_data=train_y.size(0))

        # Create data loader for mini-batching
        data_loader = DataLoader(
            TensorDataset(train_x, train_y),
            **(_get_defaults('batch') | batch_kw)
        )

        # Start training loop
        for n in range(n_epochs):
            # var_optim.
            for batch_x, batch_y in data_loader:
                # Zero grad
                for optimizer in optimizers:
                    optimizer.zero_grad()

                # Call
                pred_f = self(batch_x)
                pred_g = self.noise_gp(batch_x)
                loss = - mll(pred_f, pred_g, batch_y)

                # Backward and propr
                loss.backward()
                for optimizer in optimizers:
                    optimizer.step()

            # if n == 0 or (n + 1) % 25 == 0 and verbose:
            if verbose:
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
            pred_f = self(test_x)
            pred_g = self.noise_gp(test_x)
            # mean, covar = f_dist.mean, f_dist.lazy_covariance_matrix
            # noise_covar = self.noise_gp.added_noise(test_x)
            # y_obs = MultivariateNormal(mean, covar + noise_covar)
            y_obs = self.likelihood(pred_f, pred_g)

            lower, upper = y_obs.confidence_region()

        if return_ci:
            return pred_f, lower, upper

        return pred_f

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
