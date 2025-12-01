# Imports
import jax
import jax.numpy as jnp
import numpy as np
from jax import config
config.update("jax_enable_x64", True)




# Bimodal Gaussian
def make_bimodal_gaussian_logdensity(mean1, mean2, correlation=0.9):
    
    # Choose the two means
    mean1 = jnp.asarray(mean1)
    mean2 = jnp.asarray(mean2)
    
    # Detect how many dimensions
    # Just please don't give means with different dimensions
    dim = mean1.shape[0]

    # Make covariance matrix
    # Put correlation everywhere, add in 1 - correlation on the diagonal
    # 1s on diagonal, rho elsewhere
    cov = jnp.eye(dim) * (1 - correlation) + jnp.ones((dim, dim)) * correlation
    
    # Get inverse
    cov_inv = jnp.linalg.inv(cov)
    
    # Get log of determinant of the covariance matrix
    log_det = jnp.linalg.slogdet(cov)[1]

    # Put it all together
    def logdensity(x):
        
        # (x - mu_1) and (x - mu_2)
        diff1 = x - mean1
        diff2 = x - mean2

        # Work out what the log of a Normal looks like
        log_p1 = -0.5 * (diff1 @ cov_inv @ diff1 + log_det + dim * jnp.log(2 * jnp.pi))
        log_p2 = -0.5 * (diff2 @ cov_inv @ diff2 + log_det + dim * jnp.log(2 * jnp.pi))

        # log(0.5 e^log(N(mu1, cov)) + 0.5 e^log(N(mu2, cov)))
        return jnp.log(0.5) + jnp.logaddexp(log_p1, log_p2)

    return logdensity




# Make a correlated Gaussian
def make_correlated_gaussian_logdensity(dim, correlation = 0.5):
    
    # Make covariance matrix just like the bimodal density above
    cov = jnp.eye(dim) * (1 - correlation) + jnp.ones((dim, dim)) * correlation
    
    # Inverse of cov
    cov_inv = jnp.linalg.inv(cov)
    
    # Log det cov
    log_det = jnp.linalg.slogdet(cov)[1]

    def logdensity(x):
        # log density is prop to      -0.5 x^T cov^-1 x      -0.5 log(det(cov)) - dim/2 log(2 pi)
        return -0.5 * (x @ cov_inv @ x + log_det + dim * jnp.log(2 * jnp.pi))

    return logdensity




# Make a banana density
def make_banana_logdensity(dim, a = 1.0, b = 100):
    
    # Rosenbrock/crescent/banana
    # https://en.wikipedia.org/wiki/Rosenbrock_function#Multidimensional_generalizations
        
    # Here has the log density
    # https://uqpyproject.readthedocs.io/en/stable/auto_examples/sampling/mcmc/mcmc_metropolis_hastings.html
    
    def logdensity(x):
        
        # add up log probabilities
        total = 0.0
        
        
        for i in range(dim - 1):  # 0, 1, ..., dim - 2           so dim -1 more directions here
            
            total += -(a - x[i])**2
            total += -b * (x[i+1] - x[i]**2)**2         # i+1 here 
            
        return total
    return logdensity

