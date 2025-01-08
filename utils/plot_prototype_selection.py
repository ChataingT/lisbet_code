"""
Prototype selection
===================

Visualise the results of the prototype selection step.

We use the data from the CalMS21 Task1 dataset (Sun et al., 2021).
"""

# %%
# Import and configure modules
# ----------------------------
# Import the necessary modules and configure the plotting settings.
from pathlib import Path

import lisbet.plotting as betp
import matplotlib.pyplot as plt
import numpy as np
import pooch
from scipy.spatial.distance import squareform
from huggingface_hub import hf_hub_download
import argparse
# Configure
plt.rcParams['figure.constrained_layout.use'] = True

# %%
# Fetch the sample data from HuggingFace
# --------------------------------------
# Fetch and load the information file containing the results of the prototype selection.
# This file is generated by the command ``betman prototype_selection``.
def main(args=None):
    parser = argparse.ArgumentParser(description="Visualise prototype selection results.")
    parser.add_argument(
        "data_path",
        type=str,
        help="Path to the .npz file containing the prototype selection results."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="prototype_selection.png",
        help="Output path for the prototype selection plot."
    )
    args = parser.parse_args(args)
    data_path = args.data_path
    hmm_info = np.load(data_path)

    # %%
    # Visualise the data
    # ------------------
    # Plot the silhouette profile and the prototype selection results.
    fig, axs = plt.subplots(
        nrows=2,
        ncols=2,
        width_ratios=[2, 16],
        height_ratios=[4, 16],
        figsize=(8, 6),
    )

    # Share axes
    axs[0, 0].sharey(axs[0, 1])
    axs[0, 1].sharex(axs[1, 1])

    # Customize layout
    fig.align_ylabels(axs[:, 0])

    betp.plot_slh_score(
        hmm_info["all_n_clusters"],
        hmm_info["all_score"],
        hmm_info["best_n_clusters"],
        hmm_info["best_score"],
        axs[0, 0],
    )

    betp.plot_slh_profile(
        distance=squareform(hmm_info["cond_dist_matrix"]),
        link_matrix=hmm_info["link_matrix"],
        cluster_labels=hmm_info["best_labels"],
        ax=axs[0, 1],
    )

    betp.plot_dendrogram(
        hmm_info["link_matrix"],
        cluster_labels=hmm_info["best_labels"],
        ax=axs[1, 0],
    )

    betp.plot_heatmap(
        squareform(hmm_info["cond_dist_matrix"]),
        hmm_info["link_matrix"],
        hmm_info["best_labels"],
        hmm_info["prototypes"],
        ax=axs[1, 1],
    )

    # Finalize plot
    axs[0, 0].legend(frameon=False)
    axs[1, 1].legend(frameon=False)
    plt.savefig(args.output_path)

if __name__ == "__main__":
    main()  