# Imports
import jax
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)
import numpy as np
import matplotlib.pyplot as plt
import time
import blackjax
import arviz as az

# I can never keep matplotlib and ggplot straight
# Matplot lib guide:
# https://matplotlib.org/stable/users/index.html

# Colours: https://matplotlib.org/stable/users/explain/colors/colormaps.html#sphx-glr-users-explain-colors-colormaps-py


# Axeis and subplots: https://matplotlib.org/stable/users/explain/axes/index.html




# Plot a comparsion of BayesOpt and Autotuned hyperparamters

# def plot_comparison(results_dict, target_name, save_path = None):
#     fig, axes = plt.subplots(2, 3, figsize = (18, 10))

#     methods = ["BayesOpt", "Auto-tuning"]
#     colours = ["#1B5872", "#794661"]

#     # ESS boxplot
#     ax = axes[0, 0] # Top left
#     ess_data = [results_dict[m]["ess"] for m in methods] # Pull out ess by hyperparam method
#     bp = ax.boxplot(ess_data, labels=methods, patch_artist=True) # Boxplot it
#     for b, c in zip(bp["boxes"], colours):
#         b.set_facecolor(c)
#         b.set_alpha(0.7)
#     ax.set_title("ESS Distribution")
#     ax.grid(True, alpha = 0.25)

#     # R-hat bars
#     ax = axes[0, 1] # Top row in the middle
#     rhat_vals = [results_dict[m]["rhat"] for m in methods] # Pull out the rhat values
#     ax.bar(methods, rhat_vals, color = colours, alpha = 0.8)
#     ax.set_title("Max R-hat")
#     ax.legend()
#     ax.grid(True, alpha = 0.25)

#     # Acceptance rate
#     ax = axes[0, 2]
#     acc_data = [results_dict[m]["acceptance"] for m in methods]
#     bp = ax.boxplot(acc_data, labels = methods, patch_artist = True)
#     for b, c in zip(bp["boxes"], colours):
#         b.set_facecolor(c)
#         b.set_alpha(0.7)
#     ax.set_title("Acceptance Rate")
#     ax.legend()
#     ax.grid(True, alpha = 0.25)

#     # Time for each chain
#     ax = axes[1, 0] # Bottom left
#     time_data = [results_dict[m]["time_per_chain"] for m in methods]
#     bp = ax.boxplot(time_data, labels = methods, patch_artist = True)
#     for b, c in zip(bp["boxes"], colours):
#         b.set_facecolor(c)
#         b.set_alpha(0.7)
#     ax.set_title("Time per Chain (seconds)")
#     ax.grid(True, alpha = 0.25)

#     # L values # Actually why am I plotting this?
#     ax = axes[1, 1]
#     L_data = [results_dict[m]["L"] for m in methods]
#     bp = ax.boxplot(L_data, labels = methods, patch_artist = True)
#     for b, c in zip(bp["boxes"], colours):
#         b.set_facecolor(c)
#         b.set_alpha(0.7)
#     ax.set_title("L Distribution")
#     ax.grid(True, alpha = 0.25)

#     # Summary table so I dont need to strolll through all the code output, just amke a plot, kinda janky
#     ax = axes[1, 2]
#     ax.axis("off")

#     table_data = [] # Hmmmm maybe I should add sd or var
#     for m in methods:
#         r = results_dict[m]
#         table_data.append([
#             m,
#             f"{jnp.mean(r['ess']):.0f}",
#             f"{r['rhat']:.3f}",
#             f"{jnp.mean(r['acceptance']):.2f}",
#             f"{jnp.mean(r['time_per_chain']):.1f}s"
#         ])

#     tbl = ax.table(
#         cellText=table_data,
#         colLabels=["Method", "ESS", "R-hat", "Acc", "Time"],
#         cellLoc="center",
#         loc="center"
#     )
#     tbl.auto_set_font_size(False)
#     tbl.set_fontsize(10)

#     plt.suptitle(f"MAMS Comparison – {target_name}", fontsize=16, fontweight="bold")
#     plt.tight_layout()

#     if save_path:
#         plt.savefig(save_path, dpi=150, bbox_inches="tight")

#     return fig



# Same as above but bin the L 
# And save the table as a seperate figure, yes, kinda janky

def plot_comparison(results_dict, target_name, save_path = None):
    fig, axes = plt.subplots(2, 2, figsize=(18, 10))

    methods = ["BayesOpt", "Auto-tuning"]
    colours = ["#1B5872", "#794661"] # I love the matplotlib colour boxes

    # ESS boxplot
    ax = axes[0, 0]
    ess_data = [results_dict[m]["ess"] for m in methods]
    bp = ax.boxplot(ess_data, labels = methods, patch_artist = True)
    for b, c in zip(bp["boxes"], colours):
        b.set_facecolor(c)
        b.set_alpha(0.8)
    ax.set_title("ESS Distribution")
    ax.grid(True, alpha = 0.25)

    # R-hat barplot
    ax = axes[0, 1]
    rhat_vals = [results_dict[m]["rhat"] for m in methods]
    ax.bar(methods, rhat_vals, color=colours, alpha = 0.8)
    ax.set_title("Max R-hat")
    ax.grid(True, alpha = 0.25)

    # Acceptance rate box plot
    ax = axes[1, 0]
    acc_data = [results_dict[m]["acceptance"] for m in methods]
    bp = ax.boxplot(acc_data, labels = methods, patch_artist = True)
    for b, c in zip(bp["boxes"], colours):
        b.set_facecolor(c)
        b.set_alpha(0.8)
    ax.set_title("Acceptance Rate")
    ax.grid(True, alpha = 0.25)

    # Time box plot
    ax = axes[1, 1]
    time_data = [results_dict[m]["time_per_chain"] for m in methods]
    bp = ax.boxplot(time_data, labels = methods, patch_artist = True)
    for b, c in zip(bp["boxes"], colours):
        b.set_facecolor(c)
        b.set_alpha(0.8)
    ax.set_title("Time per Chain (s)")
    ax.grid(True, alpha = 0.25)

    plt.suptitle(f"MAMS Hyperparameter Tuning Comparison: {target_name}", fontsize = 16, fontweight = "bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi = 150, bbox_inches = "tight")
        
        
        

    # Table figure, yeah kinda weird
    fig_table, ax_table = plt.subplots(figsize = (12, 4))
    ax_table.axis("off")

    table_data = []
    for m in methods:
        r = results_dict[m]
        table_data.append([ # What I want to save
            m,
            f"{jnp.mean(r['ess']):.1f} +/- sd: {jnp.std(r['ess']):.2f}",
            f"{r['rhat']:.3f}",
            f"{jnp.mean(r['acceptance']):.2f} +/- sd: {jnp.std(r['acceptance']):.2f}",
            f"{jnp.mean(r['time_per_chain']):.1f}s +/- sd: {jnp.std(r['time_per_chain']):.2f}"
        ])

    tbl = ax_table.table(
        cellText=table_data,
        colLabels=["Method", "ESS", "R-hat", "Acc", "Time"],
        cellLoc="center",
        loc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)

    plt.suptitle(f"MAMS Hyperparameter Tuning Comparison: {target_name}", fontsize=16, fontweight="bold")
    plt.tight_layout()

    if save_path:
        table_path = save_path.replace('.png', '_table.png')
        fig_table.savefig(table_path, dpi=150, bbox_inches="tight")

    return fig, fig_table



# Plot the traces for the first coordinate and make a scatter plot for twop given dimensions
# Why cant Python start at 1?
def plot_trace_and_samples(results_dict, target_name, dim_pair = (0, 1), num_chains_to_plot = None, save_path = None):
    
    methods = ['BayesOpt', 'Auto-tuning']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    dim1, dim2 = dim_pair
    
    for method_id, method in enumerate(methods):
        samples = results_dict[method]['samples']
        
        # Default to all chains if num_chains_to_plot is None
        # 10 or 50 chains might look hporrible
        num_chains = samples.shape[0] if num_chains_to_plot is None else min(num_chains_to_plot, samples.shape[0])
        
        
        # Trace plot for dimension 1 which is zero in python
        ax = axes[method_id, 0]
        for chain_id in range(num_chains):
            chain_samples = samples[chain_id, :, dim1]
            ax.plot(chain_samples, alpha = 0.5, linewidth = 1, label=f'Chain {chain_id + 1}')
        ax.set_xlabel('Iteration', fontsize = 10)
        ax.set_ylabel(f'Dimension {dim1}', fontsize = 10)
        ax.set_title(f'{method}: Trace Plot (Dim {dim1})', fontsize = 12, fontweight='bold')
        ax.grid(True, alpha = 0.25)
        if num_chains <= 5:# Legend if a managebale number of chains
            ax.legend(fontsize = 9)
        
        
        # Make a path of samples in 2d
        ax = axes[method_id, 1]
        for chain_id in range(num_chains):
            x = samples[chain_id, :, dim1]
            y = samples[chain_id, :, dim2]
            ax.scatter(x, y, s = 5, alpha = 0.5, c = range(len(x)), cmap = 'viridis', label=f'Chain {chain_id+1}' if chain_id == 0 else '')
            ax.plot(x, y, color='gray', alpha = 0.25, linewidth = 1)
        ax.set_xlabel(f'Dimension {dim1}', fontsize = 10)
        ax.set_ylabel(f'Dimension {dim2}', fontsize = 10)
        ax.set_title(f'{method}: Sample Path (Dims {dim1}, {dim2})', fontsize = 12, fontweight = 'bold')
        ax.grid(True, alpha = 0.25)
    
    plt.suptitle(f'{target_name}: Traces and Chain Paths', fontsize = 14, fontweight = 'bold', y = 0.98)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi = 150, bbox_inches = 'tight')
        print(f"Saved trace plot: {save_path}")
    
    return fig




# Plot the hyperparameter BayesOpt search

def plot_bayesopt_progress(bayes_results, target_name, save_path = None):
    
    # Pull out each thing
    it = bayes_results["iteration"]
    ess = bayes_results["ess"]
    rhat = bayes_results["rhat"]
    obj = bayes_results["objective"]
    Ls = [h["L"] for h in bayes_results["hyperparams"]]
    steps = [h["step_size"] for h in bayes_results["hyperparams"]]

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # ESS vs iteration in the top left
    ax = axes[0, 0]
    ax.plot(it, ess, "o-")
    ax.set_title("ESS over iterations")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("ESS")
    # Set integer ticks
    ax.set_xticks(range(len(it)))
    ax.set_xticklabels([i+1 for i in range(len(it))])
    ax.grid(True, alpha = 0.25)


    # R-hat vs iteration in top right
    ax = axes[0, 1]
    ax.plot(it, rhat, "o-")
    ax.axhline(1.0, color = "red", linestyle = "--", label = "Target") # Why color not colour?
    ax.set_title("R-hat over iterations")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("R-hat")
    ax.set_xticks(range(len(it)))
    ax.set_xticklabels([i+1 for i in range(len(it))])
    ax.legend()
    ax.grid(True, alpha = 0.25)


    # Hyperparameter (L, e) exploration in the bottom left
    ax = axes[1, 0]
    scatter = ax.scatter(Ls, steps, c = obj, cmap = "viridis", s = 100) # objective function to colour the points
    ax.set_xlabel("L")
    ax.set_ylabel("step_size")
    ax.set_title("Hyperparameter Search")
    plt.colorbar(scatter, ax=ax, label="Objective")
    ax.grid(True, alpha = 0.25)

    # Objective function over iteration in bottom right
    ax = axes[1, 1]
    ax.plot(it, obj, "o-")
    best = int(np.argmax(obj))
    ax.scatter(it[best], obj[best], s = 200, c = "red", marker = "*", label = "Best")
    ax.set_title("Objective Progress")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Objective")
    ax.set_xticks(range(len(it)))
    ax.set_xticklabels([i+1 for i in range(len(it))])
    ax.legend()
    ax.grid(True, alpha = 0.25)

    plt.suptitle(f"BayesOpt Hyperparameter Tuning Progress: {target_name}", fontsize = 16, fontweight = "bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi = 150, bbox_inches = "tight")

    return fig



