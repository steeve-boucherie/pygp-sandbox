"""Benchmark script comparing GPyTorch and GPJax for heteroskedastic \
    Gaussian Process regression with sparse approximations.
"""
import argparse
import logging
import time

import gpjax as gpx

import gpytorch

import jax
import jax.numpy as jnp
# from jaxtyping import Float, Array

import matplotlib.pyplot as plt

import numpy as np

import optax

from scipy import stats

import torch


# LOGGER
logger = logging.getLogger(__name__)


# UTILS
def generate_homoscedastic_data(
    n_points: int = 1000,
    t_df: int = 3,
    t_loc: float = 0.0,
    t_scale: float = 2.1,
    m: float = 3.,
    sigma: float = 0.1,
    seed: int = 42
) -> np.ndarray:
    """
    Generate synthetic heteroskedastic data.

    Notes
    -----
    - Sample x from a Student's t-distribution.
    - Use power low of cosine(x) for y true-values.
    - Apply white noise - normally distributed (iid)

    Parameters:
    -----------
    n_points : int
        Number of data points
    t_df : float
        Degrees of freedom for Student's t distribution
    t_loc : float
        Location parameter (mean) for Student's t
    t_scale : float
        Scale parameter for Student's t
    m: float
        The power exponent to apply to the cosine of x.
    sigma: float
        The standard deviation of the white noise.
    seed : int
        Random seed for reproducibility
    """
    rng = np.random.RandomState(seed)

    # Generate x from Student's t distribution
    x = stats.t.rvs(
        t_df,
        loc=t_loc,
        scale=t_scale,
        size=n_points,
        random_state=rng
    )
    x = np.sort(x)  # Sort for nicer visualization

    # True function: power cosine

    y_true = np.power(np.cos(np.deg2rad(x)), m)

    # # Heteroskedastic noise: sigma depends on x
    # # Use log(|x| + 1) to avoid issues with x near 0
    # sigma_x = 0.1 + 0.3 * np.log(np.abs(x) + 1)
    # noise = np.random.normal(0, sigma_x)

    # Homoscedastic noise
    noise = stats.norm.rvs(0, sigma, size=n_points, random_state=rng)

    # Add heteroskedastic noise
    y = y_true + noise

    return x, y, y_true  # , sigma_x


def benchmark_gpytorch(x_train, y_train, x_test, n_inducing=100):
    """
    Fit sparse variational GP using GPyTorch with heteroskedastic noise.
    """
    print("\n" + "="*50)
    print("GPyTorch Benchmark")
    print("="*50)

    # Convert to torch tensors
    x_train_torch = torch.FloatTensor(x_train).reshape(-1, 1)
    y_train_torch = torch.FloatTensor(y_train)
    x_test_torch = torch.FloatTensor(x_test).reshape(-1, 1)

    # Define sparse variational GP model
    class SVGPModel(gpytorch.models.ApproximateGP):
        def __init__(self, inducing_points):
            variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
                inducing_points.size(0)
            )
            variational_strategy = gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution,
                learn_inducing_locations=True
            )
            super().__init__(variational_strategy)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel()
            )

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    # Initialize inducing points (uniformly across data range)
    inducing_points = torch.linspace(x_train.min(), x_train.max(), n_inducing).reshape(-1, 1)

    # Initialize model and likelihood
    model = SVGPModel(inducing_points)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    # Training mode
    model.train()
    likelihood.train()

    # Optimizer
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
        {'params': likelihood.parameters()},
    ], lr=0.01)

    # Loss function (variational ELBO)
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=y_train_torch.size(0))

    # Training
    n_iterations = 500
    start_time = time.time()

    for i in range(n_iterations):
        optimizer.zero_grad()
        output = model(x_train_torch)
        loss = -mll(output, y_train_torch)
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f"Iteration {i+1}/{n_iterations}, Loss: {loss.item():.4f}")

    training_time = time.time() - start_time

    # Prediction
    model.eval()
    likelihood.eval()

    start_time = time.time()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        predictions = likelihood(model(x_test_torch))
        mean = predictions.mean.numpy()
        lower, upper = predictions.confidence_region()
        lower = lower.numpy()
        upper = upper.numpy()

    prediction_time = time.time() - start_time

    print(f"\nTraining time: {training_time:.2f}s")
    print(f"Prediction time: {prediction_time:.4f}s")

    return mean, lower, upper, training_time, prediction_time


def benchmark_gpjax(x_train, y_train, x_test, n_inducing=100):
    """
    Fit sparse variational GP using GPJax with heteroskedastic noise.
    """
    print("\n" + "="*50)
    print("GPJax Benchmark")
    print("="*50)

    # Convert to JAX arrays
    x_train_jax = jnp.array(x_train).reshape(-1, 1)
    y_train_jax = jnp.array(y_train).reshape(-1, 1)
    x_test_jax = jnp.array(x_test).reshape(-1, 1)

    # Create dataset
    D = gpx.Dataset(X=x_train_jax, y=y_train_jax)

    # Initialize inducing points
    z = jnp.linspace(x_train.min(), x_train.max(), n_inducing).reshape(-1, 1)

    # Define prior
    kernel = gpx.kernels.RBF()
    mean_function = gpx.mean_functions.Constant()
    prior = gpx.gps.Prior(mean_function=mean_function, kernel=kernel)

    # Define likelihood (Gaussian)
    likelihood = gpx.likelihoods.Gaussian(num_datapoints=D.n)

    # Create SVGP posterior
    posterior = prior * likelihood

    # Variational family (with inducing points)
    variational_family = gpx.variational_families.VariationalGaussian(
        posterior=posterior,
        inducing_inputs=z
    )

    # ELBO objective
    elbo = gpx.objectives.ELBO(negative=True)

    # Optimize
    optimizer = optax.adam(learning_rate=0.01)

    # Training
    n_iterations = 500

    # JIT compile the training step
    @jax.jit
    def step(params, opt_state, D):
        loss_val, grads = jax.value_and_grad(
            lambda p: elbo(p, variational_family, D)
        )(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_val

    # Initialize parameters
    params = variational_family.init_params(jax.random.PRNGKey(42))
    opt_state = optimizer.init(params)

    start_time = time.time()

    for i in range(n_iterations):
        params, opt_state, loss = step(params, opt_state, D)

        if (i + 1) % 100 == 0:
            print(f"Iteration {i+1}/{n_iterations}, Loss: {float(loss):.4f}")

    training_time = time.time() - start_time

    # Prediction
    start_time = time.time()

    # Get predictive distribution
    latent_dist = variational_family(params, x_test_jax)
    predictive_dist = likelihood(params, latent_dist)

    mean = predictive_dist.mean()
    std = predictive_dist.stddev()

    mean_np = np.array(mean).flatten()
    lower_np = np.array(mean - 1.96 * std).flatten()
    upper_np = np.array(mean + 1.96 * std).flatten()

    prediction_time = time.time() - start_time

    print(f"\nTraining time: {training_time:.2f}s")
    print(f"Prediction time: {prediction_time:.4f}s")

    return mean_np, lower_np, upper_np, training_time, prediction_time


def plot_results(
    x_train,
    y_train,
    x_test,
    y_true,
    mean_gpytorch,
    lower_gpytorch,
    upper_gpytorch,
    mean_gpjax,
    lower_gpjax,
    upper_gpjax
) -> None:
    """
    Plot comparison of both methods.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # GPyTorch results
    axes[0].scatter(x_train, y_train, alpha=0.3, s=10, label='Training data')
    axes[0].plot(x_test, y_true, 'r-', linewidth=2, label='True function', alpha=0.7)
    axes[0].plot(x_test, mean_gpytorch, 'b-', linewidth=2, label='GP mean')
    axes[0].fill_between(
        x_test, lower_gpytorch, upper_gpytorch,
        alpha=0.3, label='95% confidence')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].set_title('GPyTorch (Sparse Variational GP)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # GPJax results
    axes[1].scatter(x_train, y_train, alpha=0.3, s=10, label='Training data')
    axes[1].plot(x_test, y_true, 'r-', linewidth=2, label='True function', alpha=0.7)
    axes[1].plot(x_test, mean_gpjax, 'g-', linewidth=2, label='GP mean')
    axes[1].fill_between(
        x_test, lower_gpjax, upper_gpjax,
        alpha=0.3, color='green', label='95% confidence')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    axes[1].set_title('GPJax (Sparse Variational GP)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('gp_comparison.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved as 'gp_comparison.png'")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Benchmark GPyTorch vs GPJax')
    parser.add_argument('--n_points', type=int, default=1000,
                        help='Number of training points')
    parser.add_argument('--t_df', type=float, default=3.0,
                        help='Degrees of freedom for Student t')
    parser.add_argument('--t_loc', type=float, default=0.0,
                        help='Location (mean) for Student t')
    parser.add_argument('--t_scale', type=float, default=2.1,
                        help='Scale parameter for Student t')
    parser.add_argument('--n_inducing', type=int, default=100,
                        help='Number of inducing points')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    print("Generating synthetic data...")
    x_train, y_train, y_true_train = generate_homoscedastic_data(
        n_points=args.n_points,
        t_df=args.t_df,
        t_loc=args.t_loc,
        t_scale=args.t_scale,
        seed=args.seed
    )

    # Create test points for prediction
    x_test = np.linspace(x_train.min(), x_train.max(), 200)
    y_true_test = np.power(np.abs(x_test), 0.7) * np.cos(2 * x_test)

    print(f"\nDataset size: {len(x_train)} points")
    print(f"X range: [{x_train.min():.2f}, {x_train.max():.2f}]")
    print(f"Number of inducing points: {args.n_inducing}")

    # Benchmark GPyTorch
    mean_gpytorch, lower_gpytorch, upper_gpytorch, train_time_gpytorch, pred_time_gpytorch = \
        benchmark_gpytorch(x_train, y_train, x_test, n_inducing=args.n_inducing)

    # Benchmark GPJax
    mean_gpjax, lower_gpjax, upper_gpjax, train_time_gpjax, pred_time_gpjax = \
        benchmark_gpjax(x_train, y_train, x_test, n_inducing=args.n_inducing)

    # Print summary
    print("\n" + "="*50)
    print("BENCHMARK SUMMARY")
    print("="*50)
    print(f"{'Method':<15} {'Training (s)':<15} {'Prediction (s)':<15}")
    print("-"*50)
    print(f"{'GPyTorch':<15} {train_time_gpytorch:<15.2f} {pred_time_gpytorch:<15.4f}")
    print(f"{'GPJax':<15} {train_time_gpjax:<15.2f} {pred_time_gpjax:<15.4f}")
    print("-"*50)
    print(
        f"{'Speedup':<15}"
        f"{train_time_gpytorch/train_time_gpjax:<15.2f}"
        f"x {pred_time_gpytorch/pred_time_gpjax:<15.2f}x"
    )

    # Plot results
    plot_results(x_train, y_train, x_test, y_true_test,
                 mean_gpytorch, lower_gpytorch, upper_gpytorch,
                 mean_gpjax, lower_gpjax, upper_gpjax)


if __name__ == "__main__":
    main()
