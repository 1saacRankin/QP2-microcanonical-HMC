import jax
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True) #################### Boax needs this x64              https://boax.readthedocs.io/en/latest/guides/Getting_Started.html

import numpy as np
import matplotlib.pyplot as plt
import time

import blackjax
import arviz as az

# Bayesian Optimization
# https://boax.readthedocs.io/en/latest/
# https://github.com/Lando-L/boax
from boax.experiments import optimization
from boax.acquisitions.surrogates import single_task_gaussian_process
from boax.acquisitions import upper_confidence_bound

# Import the slightly modified adaptation function
from modified_mams_adaptation import adjusted_mclmc_find_L_and_step_size

# Import samplers
from samplers import compute_ess, compute_rhat
from samplers import run_mams_fixed, run_mams_auto, run_multiple_chains_fixed, run_multiple_chains_auto




# Objective function for BayesOpt

def objective_function_with_convergence(ess, rhat, acceptance_rate, chain_length, num_chains = 1, rhat_threshold = 1.0):
    
    # If R-hat sucks, penalize alot
    # If R-hat > 1, take amount above 1, then penalize more for more chains and longer chains 
    # chain length * number of chains * amount R-hat is above 1
    # So smaller penalty for R-hat just above 1 but huge penalty for terrible R-hat
    penalty_rhat = chain_length * num_chains * jnp.maximum(rhat - rhat_threshold, 0)

    
    # ESS has weird behaviour when no samples are accepted
    # I dont want this
    # If the accpetance rate is too low, penalize a lot
    # If acceptenace rate is below say 0.25, it's bad, but less bad closer to 0.25
    # chain length * number of chains * amount acceptance rate is below 0.25
    # No penalty above 0.25 acceptance rate
    if acceptance_rate < 0.25:
        penalty_acc = chain_length * num_chains * (0.25 - acceptance_rate)
    else:
        penalty_acc = 0.0

    return ess - penalty_rhat - penalty_acc




# Bayes opt procedure
def run_bayesopt_tuning(logdensity_fn, initial_position, tuning_key, num_iterations = 20, tuning_chains = 5, chain_length = 1000):
    # Bayesian Optimization of (L, step_size) for MAMS.
    # For (L, step_size), run 5 chains of length 1000
    # This ensures that the chains are actually converging (big Rhat if not)
    # Note: experiment_results = [(params, obj)] is what Boax wants, only the current iteration's result
    
    
    # Store results in here
    results = {
        "iteration": [],
        "ess": [],
        "rhat": [],
        "acceptance": [],
        "objective": [],
        "hyperparams": [],
        "mean_time_per_chain": []
    }

    # Space to explore
    # Rectangel will do
    # It would be weird to have L = 0.5 and step_size = 4
    # Probably wont happen
    # A triangle would be better
    parameters = [
        {"name": "L", "type": "range", "bounds": [0.5, 40.0]},
        {"name": "step_size", "type": "range", "bounds": [0.01, 4.0]}
    ]

    # Give bounds to explore
    bounds = jnp.array([p["bounds"] for p in parameters])
    
    # Use a GP surrogate
    surrogate = single_task_gaussian_process(bounds=bounds)
    
    # Use UCB for an aquisition function
    # Could use expected improvement or something else, UCB is good
    acquisition = upper_confidence_bound(bounds=bounds, beta=2.0)

    
    # Set up experiment
    experiment = optimization(
        parameters = parameters,
        batch_size = 1, # One (L, step_size) to try at a time
        surrogate = surrogate,
        acquisition = acquisition
    )

    # Initalize step and experiment results
    # https://boax.readthedocs.io/en/latest/guides/Hyperparameter_Tuning.html
    step = None
    experiment_results = []

    # For iteration in number of iterations to do
    for it in range(num_iterations):
        
        # Tell me where we are in the BayesOpt-ing
        print(f"######################################### Iteration {it+1}/{num_iterations}")

        # Aquisition function says where to try next
        step, param_list = experiment.next(step, experiment_results)
        params = param_list[0]

        # Which L and step_size to try next
        L = float(params["L"])
        step_size = float(params["step_size"])
        
        # Tell me which L and step size we're trying
        print(f"Running {tuning_chains} chains with L = {L:.3f}, and step size = {step_size:.4f}")

        # Keep it consistent
        iter_key = jax.random.fold_in(tuning_key, it)

        
        # Use the multiple chains function with these fixed hyperparaametrs
        # Save an array of results for the multiple chains
        samples, ess_array, acc_array, time_array, _ = run_multiple_chains_fixed(
            logdensity_fn,
            tuning_chains,
            chain_length,
            initial_position,
            iter_key,
            L,
            step_size
        )

        # Find mean ESS, accpetance rate, and time for these chains and these hyperparameters
        mean_ess = float(jnp.mean(ess_array))
        mean_acc = float(jnp.mean(acc_array))
        mean_time = float(jnp.mean(time_array))
        
        # Did the chaisn converge?
        rhat = compute_rhat(samples)

        # Compute objective for these results
        obj = float(objective_function_with_convergence(mean_ess, rhat, mean_acc, chain_length, num_chains = tuning_chains))

        print(f"ESS = {mean_ess:.1f}, R-hat = {rhat:.3f}, Acceptance rate = {mean_acc:.3f}, Objective function = {obj:.1f}")

        # Update Boax
        # Overwrite is correct, see https://boax.readthedocs.io/en/latest/guides/Hyperparameter_Tuning.html
        experiment_results = [(params, obj)]
        

        # Save results to plot later
        results["iteration"].append(it)
        results["ess"].append(mean_ess)
        results["rhat"].append(rhat)
        results["acceptance"].append(mean_acc)
        results["objective"].append(obj)
        results["hyperparams"].append(params)
        results["mean_time_per_chain"].append(mean_time)
    

    # Whuch iteration had the best hyperparameters?
    best_iteration = int(np.argmax(results["objective"]))
    best_params = results["hyperparams"][best_iteration]
    
    # What are the best hyperparameters?
    print("---------------------------------------BEST HYPERPARAMTEERS ACCORDING TO BAYESOPT-----------------------------------------------------")
    print(f"Best hyperparameters: L = {best_params['L']:.3f}, step size = {best_params['step_size']:.4f}")
    
    # Return the resukts to plot and the best hyperparameters to use in the validation procedure
    return results, best_params

