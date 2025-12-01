########## Imports

# This is very much JAX
import jax
import jax.numpy as jnp

# Need to make it 64
from jax import config
config.update("jax_enable_x64", True)

# Time and plots and numpy
import numpy as np
import matplotlib.pyplot as plt
import time

# BlackJax has the samplers
import blackjax

# MCMC plots
import arviz as az

# Bayesian Optimization
from boax.experiments import optimization
from boax.acquisitions.surrogates import single_task_gaussian_process
from boax.acquisitions import upper_confidence_bound

# Import the slightly modified adaptation function
from modified_mams_adaptation import adjusted_mclmc_find_L_and_step_size

# Import target densities
from target_densities import make_bimodal_gaussian_logdensity, make_correlated_gaussian_logdensity, make_banana_logdensity

# Import plot functions
from plot_functions import plot_comparison, plot_trace_and_samples, plot_bayesopt_progress

# Import BayesOpt 
from bayes_opt import objective_function_with_convergence, run_bayesopt_tuning

# Import samplers (and ess and rhat)
from samplers import compute_ess, compute_rhat
from samplers import run_mams_fixed, run_mams_auto, run_multiple_chains_fixed, run_multiple_chains_auto




# Run a few longer chains with the BayesOpt-ed hyperparameters from a few positions
# Run autotuned chains as well
# See how they do
def validate_method(method_name, logdensity_fn, initial_position, validation_key, num_chains = 10, num_steps = 2000, L = None, step_size = None, tune_mass_matrix=False):
    
    # Run validation chains for bayesopt or autotuned
    print(f"######################################### VALIDATING: {method_name}")
    print(f"Running: {num_chains} chains each of length {num_steps}")

    # See how long it takes
    start_time = time.time()

    # If we're validating the BayesOpt-ed L and step size
    if method_name == "BayesOpt":
        
        # Use the optimized L and step_size
        print(f"Using hyperparameters: L = {L:.3f}, step_size = {step_size:.5f}")

        # Use the function to run multiple chains with fixed hyperparameters
        samples, ess_arr, acc_arr, time_arr, starts = run_multiple_chains_fixed(logdensity_fn, num_chains, num_steps, initial_position, validation_key, L, step_size)

        # Match the shape well get from the autotuned validation to see which values of L and step_size get used
        # Here it's obviosuly the BayesOpted L and step_size
        L_values = jnp.full(num_chains, L)
        step_values = jnp.full(num_chains, step_size)
    
    
    else:  # If method is not BayesOpt then it's Auto-tune
        print("######################################### Running Auto Tuning")
        print(f"Running: {num_chains} chains each of length {num_steps}")
        samples, ess_arr, acc_arr, step_vals, L_vals, time_arr, starts, _ = (run_multiple_chains_auto(logdensity_fn, num_chains, num_steps, initial_position,
                validation_key, # Keep things consistent
                validation_key,  # Use same base key for the random (but consistent) starting positions
                tune_mass_matrix = tune_mass_matrix # I'll most likely always have this False
            ))

        # See which values the chains used
        step_values = step_vals
        L_values = L_vals

    # Find R-hat for the num_chains
    rhat = compute_rhat(samples)
    
    # Stop the clock
    elapsed = time.time() - start_time

    
    # Print the results, line stuff up even though the names are different lengths
    print("")
    print(f"Validation results for: {method_name}")
    print(f"            ESS mean +/- sd: {jnp.mean(ess_arr):.1f} ± {jnp.std(ess_arr):.1f}")
    print(f"                  R-hat max: {rhat:.4f}")
    print(f"Acceptance rate mean +/- sd: {jnp.mean(acc_arr):.3f} ± {jnp.std(acc_arr):.3f}")
    print(f"              L mean +/- sd: {jnp.mean(L_values):.3f} ± {jnp.std(L_values):.3f}")
    print(f"      Step size mean +/- sd: {jnp.mean(step_values):.4f} ± {jnp.std(step_values):.4f}")
    print(f" Time per chain mean +/- sd: {jnp.mean(time_arr):.2f}s ± {jnp.std(time_arr):.2f}s")
    print(f"                 Total time: {elapsed:.1f}s")

    
    # Return everytging
    return {
        "samples": samples,
        "ess": ess_arr,
        "rhat": rhat,
        "acceptance": acc_arr,
        "L": L_values,
        "step_size": step_values,
        "time_per_chain": time_arr,
        "total_time": elapsed,
        "start_positions": starts
    }




# Compare RCS's auto tuning to BayesOpt tuning on a target density
# Just turn off the preconditioner/inverse mass matrix tuning for auto tuning
def compare_methods_on_target(target_name, logdensity_fn, dim, initial_position, plot_dims = (0, 1), tune_mass_matrix = False):

    
    # Which desnity we're sampling from
    print(f"################################################## Target Densiity: {target_name}")

    
    
    # BayesOpt-it
    tuning_key = jax.random.key(548) # Stat 548 you know
    bayes_results, best_params = run_bayesopt_tuning(
        logdensity_fn,
        initial_position,
        tuning_key,
        num_iterations = 20, # Can mess around with how many BayesOpt iterations and number of chains and length of the chains
        tuning_chains = 5,
        chain_length = 1000
    )

    # Plot the BayesOpt exploration-exploitation and save it
    file_path = f"plots/bayesopt_progress_{target_name}.png"
    plot_bayesopt_progress(bayes_results, target_name, save_path=file_path)

    
    # Run many longer chains with the BayesOpt-ed hyperparameters (validate)
    val_key = jax.random.key(2025)
    val_bayes = validate_method(
        "BayesOpt",
        logdensity_fn,
        initial_position,
        val_key,
        num_chains = 10,
        num_steps = 2000,
        L = float(best_params["L"]), # Pull out the best L
        step_size = float(best_params["step_size"]) # And the best step size from BayesOpt
    )

    # Run a bucnh of auto-tuned chains from the same perturbed starting positions
    val_auto = validate_method(
        "Auto-tuning",
        logdensity_fn,
        initial_position,
        val_key, # Same key as for bayes opt
        num_chains = 10,
        num_steps = 2000,
        tune_mass_matrix = tune_mass_matrix # Probably always False
    )

    # Make a comparison plot for BayesOpt vs AutoTuned, time, ess, etc
    results = {
        "BayesOpt": val_bayes,
        "Auto-tuning": val_auto
    }
    # Save it
    comparison_path = f"plots/comparison_{target_name}.png"
    plot_comparison(results, target_name, save_path = comparison_path)


    # Plot the trace for the first dimension and a scatterplot of samples for the two given dimensions
    scatter_path = f"plots/traces_{target_name}.png"
    plot_trace_and_samples(results, target_name, dim_pair=plot_dims, 
                          num_chains_to_plot = None,  # None = plot all chains, high dimensions would look nasty if all are plotted
                          save_path = scatter_path)

    
    
    
    
    
    # Tell me what the word is on ESS
    print("##############################################################################")
    print("Results")
    print(f"BayesOpt ESS mean: {jnp.mean(val_bayes['ess']):.1f}")
    print(f".   Auto ESS mean: {jnp.mean(val_auto['ess']):.1f}")

    # Return results
    return bayes_results, results













# Run these experiemnst on the three targets: correlated Gaussian, bimodal isotropic Gaussians, Banana

# Dimension of target density
dim = 3

# Start at the origin
initial_position = jnp.zeros(dim)



# Bimodal
target1 = make_bimodal_gaussian_logdensity(mean1 = [-2] * dim, mean2 = [2] * dim, correlation = 0)
compare_methods_on_target("Bimodal Gaussian", target1, dim, initial_position, tune_mass_matrix = False)


# Correlated Gaussian
target2 = make_correlated_gaussian_logdensity(dim, correlation = 0.8)
compare_methods_on_target("Correlated Gaussian", target2, dim, initial_position, tune_mass_matrix = False)


# Banana
target3 = make_banana_logdensity(dim)
compare_methods_on_target("Banana", target3, dim, initial_position, tune_mass_matrix = False)
