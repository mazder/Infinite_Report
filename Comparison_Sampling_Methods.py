import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import time

# For Inverse Transformation
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d

# Parameters for the Gaussian distribution
mu = 1  # Mean (initial value)
sigma = 4  # Standard deviation (initial value)
alpha = 2  # Parameter for L(x)
beta = 1  # Scaling factor for L(x)
num_samples = 1000  # Number of samples to generate

proposal_std = 1.0  # Standard deviation for the proposal distribution for Metropolis-Hastings

# Calculate function L(x)
def calculate_L(x, alpha, beta):
    return np.where(x < alpha, 0, beta * (x - 1))

# Calculate the exponential term modified by L(x)
def calculate_exp_term(x, mu, sigma, alpha, beta):
    return np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) - calculate_L(x, alpha, beta)

# Calculate the normalization constant Z
def calculate_constant_Z(mu, sigma, alpha, beta):
    z, error = quad(calculate_exp_term, -np.inf, np.inf, args=(mu, sigma, alpha, beta))
    return z

# Target distribution function (normalized)
def target_distribution(x, mu, sigma, alpha, beta, z):
    return calculate_exp_term(x, mu, sigma, alpha, beta) / z

# Rejection Sampling
def accept_reject_sample(mu, sigma, alpha, beta, z, max_iterations=1000):
    for _ in range(max_iterations):
        x = np.random.normal(mu, sigma)
        p_x = target_distribution(x, mu, sigma, alpha, beta, z)
        u = np.random.uniform(0, 1)
        if p_x > 0 and u <= p_x:
            return x
    raise ValueError("Failed to generate a valid sample from the distribution after max iterations.")

def rejection_sampling(mu, sigma, alpha, beta, num_samples, max_iterations=1000):
    x_samples = []
    z = calculate_constant_Z(mu, sigma, alpha, beta)
    while len(x_samples) < num_samples:
        x = accept_reject_sample(mu, sigma, alpha, beta, z)
        x_samples.append(x)
    return np.array(x_samples)

# Metropolis-Hastings Sampling
def metropolis_hastings_sampling(mu, sigma, alpha, beta, num_samples, proposal_std):
    x_samples = []
    z = calculate_constant_Z(mu, sigma, alpha, beta)
    x_current = np.random.normal(mu, sigma)
    while len(x_samples) < num_samples:
        x_proposal = np.random.normal(x_current, proposal_std)
        p_current = target_distribution(x_current, mu, sigma, alpha, beta, z)
        p_proposal = target_distribution(x_proposal, mu, sigma, alpha, beta, z)
        acceptance_ratio = min(1, p_proposal / p_current)
        if np.random.uniform(0, 1) < acceptance_ratio:
            x_current = x_proposal
        x_samples.append(x_current)
    return np.array(x_samples)

# Gibbs Sampling
def generate_sample_g_y_given_x(x):
    mean_y = x
    sigma_y = 1
    return np.random.normal(mean_y, sigma_y)

def update_parameters(x, y, alpha, beta, mu, sigma):
    alpha = np.mean(x) + 0.1
    beta = np.std(x)
    mu = np.mean(y)
    sigma = np.std(y) + 1
    return alpha, beta, mu, sigma

def gibbs_sampling(mu, sigma, alpha, beta, num_samples, max_iterations=1000):
    x_samples = []
    y_samples = []
    z = calculate_constant_Z(mu, sigma, alpha, beta)
    while len(x_samples) < num_samples:
        xi = accept_reject_sample(mu, sigma, alpha, beta, z)
        yi = generate_sample_g_y_given_x(xi)
        alpha, beta, mu, sigma = update_parameters([xi], [yi], alpha, beta, mu, sigma)
        x_samples.append(xi)
        y_samples.append(yi)
    return np.array(x_samples)

# Inverse Transform Sampling
def inverse_transform_sampling(mu, sigma, alpha, beta, num_samples):
    z = calculate_constant_Z(mu, sigma, alpha, beta)
    x_values = np.linspace(-10, 10, num_samples)
    pdf_values = target_distribution(x_values, mu, sigma, alpha, beta, z)
    cdf_values = cumulative_trapezoid(pdf_values, x_values, initial=0)
    cdf_values /= cdf_values[-1]
    inverse_cdf = interp1d(cdf_values, x_values, bounds_error=False, fill_value="extrapolate")
    uniform_samples = np.random.uniform(0, 1, num_samples)
    x_samples = inverse_cdf(uniform_samples)
    return np.array(x_samples)


# Standard Gaussian Sampling using numpy
def standard_gaussian_sampling(mu, sigma, num_samples):
    return np.random.normal(mu, sigma, num_samples)


# Measure and print runtime for each sampling method
def measure_runtime(sampling_function, *args):
    start_time = time.time()
    samples = sampling_function(*args)
    end_time = time.time()
    runtime = end_time - start_time
    return samples, runtime

estimated_E = []
runtimes = []

# Generate samples and measure runtime using different methods
rx_samples, rx_runtime = measure_runtime(rejection_sampling, mu, sigma, alpha, beta, num_samples)
estimated_E.append(np.mean(rx_samples))
runtimes.append(rx_runtime)
mx_samples, mx_runtime = measure_runtime(metropolis_hastings_sampling, mu, sigma, alpha, beta, num_samples, proposal_std)
estimated_E.append(np.mean(mx_samples))
runtimes.append(mx_runtime)
gx_samples, gx_runtime = measure_runtime(gibbs_sampling, mu, sigma, alpha, beta, num_samples)
estimated_E.append(np.mean(gx_samples))
runtimes.append(gx_runtime)

ix_samples, ix_runtime = measure_runtime(inverse_transform_sampling, mu, sigma, alpha, beta, num_samples)
estimated_E.append(np.mean(ix_samples))
runtimes.append(ix_runtime)

sx_samples, sx_runtime = measure_runtime(standard_gaussian_sampling, mu, sigma, num_samples)
estimated_E.append(np.mean(sx_samples))
runtimes.append(sx_runtime)


# Print estimated E[x] and runtime for each method
print(f"Rejection Sampling: Estimated E[x]: {np.mean(rx_samples):.5f}, Runtime: {rx_runtime:.5f} seconds")
print(f"Metropolis-Hastings Sampling: Estimated E[x]: {np.mean(mx_samples):.5f}, Runtime: {mx_runtime:.5f} seconds")
print(f"Gibbs Sampling: Estimated E[x]: {np.mean(gx_samples):.5f}, Runtime: {gx_runtime:.5f} seconds")
print(f"Inverse Transform Sampling: Estimated E[x]: {np.mean(ix_samples):.5f}, Runtime: {ix_runtime:.5f} seconds")
print(f"Standard Gaussian Sampling: Estimated E[x]: {np.mean(sx_samples):.5f}, Runtime: {sx_runtime:.5f} seconds")


# Measure runtime for each sampling method
sampling_methods = {
    'Rejection Sampling': rejection_sampling,
    'Metropolis-Hastings Sampling': metropolis_hastings_sampling,
    'Gibbs Sampling': gibbs_sampling,
    'Inverse Transform Sampling': inverse_transform_sampling,
    'Standard Gaussian Sampling': standard_gaussian_sampling
}

# Create subplots to visualize the distributions
fig, axes = plt.subplots(5, 2, figsize=(20, 20))

# Subplot for rejection sampling - Histogram
axes[0, 0].hist(rx_samples, bins=50, density=True, alpha=0.5, color='blue')
axes[0, 0].set_title('Rejection Sampling - Histogram')
axes[0, 0].set_xlabel('x')
axes[0, 0].set_ylabel('Density')
axes[0, 0].grid(True)

# Subplot for rejection sampling - Line Plot
axes[0, 1].plot(rx_samples, color='blue', alpha=0.7)
axes[0, 1].set_title('Rejection Sampling - Line Plot')
axes[0, 1].set_xlabel('Sample Index')
axes[0, 1].set_ylabel('Sample Value')
axes[0, 1].grid(True)

# Subplot for Metropolis-Hastings sampling - Histogram
axes[1, 0].hist(mx_samples, bins=50, density=True, alpha=0.5, color='green')
axes[1, 0].set_title('Metropolis-Hastings Sampling - Histogram')
axes[1, 0].set_xlabel('x')
axes[1, 0].set_ylabel('Density')
axes[1, 0].grid(True)

# Subplot for Metropolis-Hastings sampling - Line Plot
axes[1, 1].plot(mx_samples, color='green', alpha=0.7)
axes[1, 1].set_title('Metropolis-Hastings Sampling - Line Plot')
axes[1, 1].set_xlabel('Sample Index')
axes[1, 1].set_ylabel('Sample Value')
axes[1, 1].grid(True)

# Subplot for Gibbs sampling - Histogram
axes[2, 0].hist(gx_samples, bins=50, density=True, alpha=0.5, color='red')
axes[2, 0].set_title('Gibbs Sampling - Histogram')
axes[2, 0].set_xlabel('x')
axes[2, 0].set_ylabel('Density')
axes[2, 0].grid(True)

# Subplot for Gibbs sampling - Line Plot
axes[2, 1].plot(gx_samples, color='red', alpha=0.7)
axes[2, 1].set_title('Gibbs Sampling - Line Plot')
axes[2, 1].set_xlabel('Sample Index')
axes[2, 1].set_ylabel('Sample Value')
axes[2, 1].grid(True)

# Subplot for inverse transform sampling - Histogram
axes[3, 0].hist(ix_samples, bins=50, density=True, alpha=0.5, color='purple')
axes[3, 0].set_title('Inverse Transform Sampling - Histogram')
axes[3, 0].set_xlabel('x')
axes[3, 0].set_ylabel('Density')
axes[3, 0].grid(True)

# Subplot for inverse transform sampling - Line Plot
axes[3, 1].plot(ix_samples, color='purple', alpha=0.7)
axes[3, 1].set_title('Inverse Transform Sampling - Line Plot')
axes[3, 1].set_xlabel('Sample Index')
axes[3, 1].set_ylabel('Sample Value')
axes[3, 1].grid(True)


# Subplot for inverse transform sampling - Histogram
axes[4, 0].hist(sx_samples, bins=50, density=True, alpha=0.5, color='purple')
axes[4, 0].set_title('Standard Gaussian Sampling - Histogram')
axes[4, 0].set_xlabel('x')
axes[4, 0].set_ylabel('Density')
axes[4, 0].grid(True)

# Subplot for inverse transform sampling - Line Plot
axes[4, 1].plot(sx_samples, color='purple', alpha=0.7)
axes[4, 1].set_title('Standard Gaussian Sampling - Line Plot')
axes[4, 1].set_xlabel('Sample Index')
axes[4, 1].set_ylabel('Sample Value')
axes[4, 1].grid(True)



# Adjust layout and show plot
plt.tight_layout()
plt.savefig("Sampling_Methods_Comparison_Histogram_and_Line.png", dpi=300)
#plt.show()


# New plot for comparing runtime for each sampling method
fig, ax = plt.subplots(figsize=(10, 6))

# Bar plot for runtime
methods = ['Rejection Sampling', 'Metropolis-Hastings Sampling', 'Gibbs Sampling', 'Inverse Transform Sampling', 'Standard Gaussian Sampling']
ax.bar(methods, runtimes, color='orange', alpha=0.7)

# Axis settings
ax.set_xlabel('Sampling Method')
ax.set_ylabel('Runtime (seconds)')
ax.set_title('Runtime Comparison for Different Sampling Methods')

# Save and display the plot
plt.tight_layout()
plt.savefig("Runtime_Comparison.png", dpi=300)

print(sampling_methods)
print(estimated_E)
print(runtimes)
