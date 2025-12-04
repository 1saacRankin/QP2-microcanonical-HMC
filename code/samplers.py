


######################### Imports
import jax
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True) # Boax needs this

import numpy as np
import matplotlib.pyplot as plt
import time

# Smaplers are from BlackJax
import blackjax

# MCMC plotting functions and diagnostics
import arviz as az


# Bayesian Optimization
from boax.experiments import optimization
from boax.acquisitions.surrogates import single_task_gaussian_process
from boax.acquisitions import upper_confidence_bound

# Import the slightly modified adaptation function
from modified_mams_adaptation import adjusted_mclmc_find_L_and_step_size
# https://blackjax-devs.github.io/blackjax/_modules/blackjax/adaptation/adjusted_mclmc_adaptation.html#adjusted_mclmc_find_L_and_step_size



# MAMS tutorial: https://blackjax-devs.github.io/sampling-book/algorithms/mclmc.html#adjusted-mclmc
# Documentation:
# https://blackjax-devs.github.io/blackjax/autoapi/blackjax/mcmc/adjusted_mclmc/index.html#
# https://blackjax-devs.github.io/blackjax/_modules/blackjax/mcmc/adjusted_mclmc.html#as_top_level_api

# Need to change the integrator to (isokinetic) velocity_verlet: https://blackjax-devs.github.io/blackjax/_modules/blackjax/mcmc/integrators.html#isokinetic_velocity_verlet




def compute_ess(samples):
    # Get the ESS for the coordinate with the smallest ESS
    # We want the maximize the Minimum ESS (maximin)
    # https://blackjax-devs.github.io/blackjax/autoapi/blackjax/diagnostics/index.html#blackjax.diagnostics.effective_sample_size
    
    # Get a vector of ESSs
    ess_vec = blackjax.diagnostics.effective_sample_size(samples)
    
    # Return the min
    return float(jnp.min(ess_vec))



def compute_rhat(chains):
    # Get the R-hat for the worst dimension
    # https://blackjax-devs.github.io/blackjax/autoapi/blackjax/diagnostics/index.html#blackjax.diagnostics.potential_scale_reduction
    # We want to minimize the maximum R-hat (minimax)
    
    # Get R-hat for each dimension
    rhat_vec = blackjax.diagnostics.potential_scale_reduction(chains)
    
    # Return the max
    return float(jnp.max(rhat_vec))








# Run MAMS with fixed hyperparameters
# Modified from here: https://blackjax-devs.github.io/sampling-book/algorithms/mclmc.html#how-to-run-mclmc-in-blackjax
def run_mams_fixed(logdensity_fn, chain_length, initial_position, key, L, step_size):
    
    # Time it
    start_time = time.time()
    
    # Needs initialization and run keys, no tuning key since given hyperparamaters
    init_key, run_key = jax.random.split(key)
    
    # Calculate num_integration_steps from L and step_size
    num_integration_steps = int(jnp.ceil(L / step_size))
    
    # Initial state
    initial_state = blackjax.mcmc.adjusted_mclmc.init(
        position=initial_position,
        logdensity_fn=logdensity_fn
    )
    
    # Create algorithm 
    algorithm = blackjax.adjusted_mclmc(
        logdensity_fn = logdensity_fn,
        step_size = step_size, # Given step size
        num_integration_steps = num_integration_steps, # Eseentialy L
        integrator = blackjax.mcmc.integrators.isokinetic_velocity_verlet # Want leapfrog/verlocit verlet
    )
    
    # Do one proposal and keep info so we can look at the accpetance rates
    def one_step(state, key):
        state, info = algorithm.step(key, state)
        return state, (state.position, info.acceptance_rate)
    
    # Keep it consistent
    keys = jax.random.split(run_key, chain_length)
    final_state, (samples, acceptance_rates) = jax.lax.scan(one_step, initial_state, keys)
    
    # For this chain what is the average acceotance, ess, and time it took?
    avg_acceptance = float(jnp.mean(acceptance_rates))
    ess = compute_ess(samples)
    elapsed = time.time() - start_time
    
    # Return that
    return samples, ess, avg_acceptance, elapsed





# Pretty much default but with tracking acceptance rate and my modified adjusted_mclmc_find_L_and_step_size
# https://blackjax-devs.github.io/sampling-book/algorithms/mclmc.html
def run_mams_auto(logdensity_fn, num_steps, initial_position, key, tune_mass_matrix = False):
    
    # Time it
    start_time = time.time()
    
    # Need tuning key for this one
    init_key, tune_key, run_key = jax.random.split(key, 3)
    
    # Initialize sampler
    initial_state = blackjax.mcmc.adjusted_mclmc.init(
        position=initial_position,
        logdensity_fn=logdensity_fn
    )
    
    # Kernel function for tuning, see sampling book
    def kernel(rng_key, state, avg_num_integration_steps, step_size, inverse_mass_matrix):
        num_steps_int = jnp.ceil(avg_num_integration_steps).astype(int)
        
        kernel_fn = blackjax.mcmc.adjusted_mclmc.build_kernel(
            logdensity_fn = logdensity_fn,
            inverse_mass_matrix = inverse_mass_matrix,
            integrator = blackjax.mcmc.integrators.isokinetic_velocity_verlet # Leapfrog
        )
        
        return kernel_fn(
            rng_key=rng_key,
            state=state,
            step_size=step_size,
            num_integration_steps=num_steps_int,
        )
    
    
    # Swap out the BlackJax adjusted_mclmc_find_L_and_step_size for the modified one
    #(state_after_tuning, sampler_params, _) = blackjax.adjusted_mclmc_find_L_and_step_size(
    (state_after_tuning, sampler_params, _) = adjusted_mclmc_find_L_and_step_size(
        mclmc_kernel=kernel,
        num_steps=num_steps,
        state=initial_state,
        rng_key=tune_key,
        target=0.90,  # RCS say 0.90
        frac_tune1=0.1, # step size with L fixed now
        frac_tune2=0.0, # Make it 0.1 if I want to tune the mass matrix
        frac_tune3=0.1, # L with step size fixed
        diagonal_preconditioning = tune_mass_matrix, # Probably always False
        fix_L_first_da=True ################### New in my function
    )
    
    # Pull out the tuned parameters
    step_size = sampler_params.step_size
    L = sampler_params.L
    inverse_mass_matrix = sampler_params.inverse_mass_matrix
    print(f"Auto tuning chose: L = {L:.3f} and step size = {step_size:.5f} and inverse mass matrix = {inverse_mass_matrix}")

    
    # Calculate num_integration_steps
    num_integration_steps = int(jnp.ceil(L / step_size))
    
    # Run sampling with tuned parameters
    alg = blackjax.adjusted_mclmc(
        logdensity_fn = logdensity_fn,
        step_size = step_size,
        num_integration_steps = num_integration_steps,
        inverse_mass_matrix = inverse_mass_matrix,
        integrator = blackjax.mcmc.integrators.isokinetic_velocity_verlet # Leapfrog
    )
    
    # Get the info for one proposal
    def one_step(state, key):
        state, info = alg.step(key, state)
        return state, (state.position, info.acceptance_rate)
    
    # Keep it consitent
    keys = jax.random.split(run_key, num_steps)
    
    # Run the chain
    final_state, (samples, acceptance_rates) = jax.lax.scan(one_step, state_after_tuning, keys)
    
    # What is the average accpetance rate and ESS?
    avg_acceptance = float(jnp.mean(acceptance_rates))
    ess = compute_ess(samples)
    
    # Compute the time elpased
    elapsed = time.time() - start_time
    
    # Return what I need
    return samples, step_size, L, inverse_mass_matrix, avg_acceptance, ess, elapsed










# Run a few chains with given hyperparameters
def run_multiple_chains_fixed(logdensity_fn, num_chains, num_steps, initial_position, base_key, L, step_size):

    # Storage for everything
    all_samples = []
    all_ess = []
    all_accept = []
    all_times = []
    all_start = []

    # Run the num_chains
    for i in range(num_chains):
        
        # Keep each chain reproducible but make sure tehre's randomness between chains
        chain_key = jax.random.fold_in(base_key, i)
        perturb_key = jax.random.fold_in(base_key, i + 1)

        # Give each chain a different starting position
        start_pos = initial_position + 1.0 * jax.random.normal(perturb_key, initial_position.shape)
        all_start.append(start_pos)

        print(f"---------------- Chain {i+1} started at {start_pos[:5]} (first 5 coordinates)")

        # use the function from above
        samples, ess, acc, t = run_mams_fixed(
            logdensity_fn,
            chain_length = num_steps, # num_steps in the chain, not leapfrog steps, bad name, my bad
            initial_position = start_pos,
            key = chain_key,
            L = L,
            step_size = step_size
        )

        # Length of chain
        num_samples = samples.shape[0]
        # Info on the chian
        print(f"------ ESS = {ess:.1f}, Acceptance rate = {acc:.3f}, Time = {t:.2f}s, Samples = {num_samples}")

        
        # Store the info
        all_samples.append(samples)
        all_ess.append(ess)
        all_accept.append(acc)
        all_times.append(t)

    
    # Stack up all the samples from all the chains
    all_samples = jnp.stack(all_samples, axis = 0)

    # Give everything back
    return (
        all_samples,
        jnp.array(all_ess),
        jnp.array(all_accept),
        jnp.array(all_times),
        all_start
    )







# Use the MAMS _auto function but now multiple times
def run_multiple_chains_auto(logdensity_fn, num_chains, num_steps, initial_position, base_key, perturb_base_key, tune_mass_matrix = False):

    
    # Store results
    all_samples = []
    all_ess = []
    all_accept = []
    all_L = []
    all_step = []
    all_times = []
    all_start = []
    all_inverse_mass_matrices = [] # If I want to tune, will it tune even though it shouldnt tune a diagonal matrix for a correlated Gaussian?

    
    # Loop over all the chains
    # All esentially the same as for the fixed mams function above
    for i in range(num_chains):
        chain_key = jax.random.fold_in(base_key, i)
        perturb_key = jax.random.fold_in(perturb_base_key, i + 1)

        start_pos = initial_position + 1.0 * jax.random.normal(perturb_key, initial_position.shape)
        all_start.append(start_pos)

        print(f"---------------- Chain {i+1} started at {start_pos[:5]} (first 5 coordinates)")

        samples, step_size, L, inverse_mass_matrix, acc, ess, t = run_mams_auto(
            logdensity_fn,
            num_steps,
            start_pos,
            key=chain_key,
            tune_mass_matrix=tune_mass_matrix
        )

        # Info on this chain
        num_samples = samples.shape[0]
        print(f"------ L = {L:.2f}, step size = {step_size:.5f}  ESS = {ess:.1f}, Acceptance rate = {acc:.3f}, Time = {t:.2f}s, Samples = {num_samples}")

        
        # Add to the results
        all_samples.append(samples)
        all_ess.append(ess)
        all_accept.append(acc)
        all_L.append(L)
        all_step.append(step_size)
        all_times.append(t)
        all_inverse_mass_matrices.append(inverse_mass_matrix)

    
    # stack all the samples
    all_samples = jnp.stack(all_samples, axis=0)

    # Give everything back 
    return (
        all_samples,
        jnp.array(all_ess),
        jnp.array(all_accept),
        jnp.array(all_step),
        jnp.array(all_L),
        jnp.array(all_times),
        all_start,
        all_inverse_mass_matrices
    )
