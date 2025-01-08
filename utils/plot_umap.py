"""
Visualize Embeddings
====================

Visualise the results of the embedding step.

We use the data from the CalMS21 Task1 dataset (Sun et al., 2021).
"""


from pathlib import Path

import lisbet.plotting as betp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lisbet.datasets import load_records
from tqdm.auto import tqdm
import argparse
import os


def main(args=None):
    # Plot configuration
    plt.rcParams['figure.constrained_layout.use'] = True

    # Argument parser
    parser = argparse.ArgumentParser(description="Visualize UMAP embeddings.")
    parser.add_argument("--behaviors", nargs='+', default=["ASD", "TD"], help="List of behaviors.")
    parser.add_argument("--umap_dim", type=int, default=2, help="UMAP dimension.")
    parser.add_argument("--umap_ngh", type=int, default=300, help="UMAP number of neighbors.")
    parser.add_argument("--umap_path", type=str, default="/home/share/schaer2/thibaut/humanlisbet/test_betman/embeddings", help="Path to UMAP embeddings.")
    parser.add_argument("--test_set_path", type=str, default="/home/share/schaer2/thibaut/humanlisbet/datasets/humans/humans_test_annoted_30.h5", help="Path to the test set.")
    parser.add_argument("--output_path", type=str, default="test_betman/umap.png", help="Output path for the UMAP plot.")
    parser.add_argument("--sample_size", type=int, default=10000, help="Sample size for UMAP plot.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed. None for umap to be used with multiple processes.")
    parser.add_argument("--data_format", type=str, default="h5archive", help="Data format.")
    args = parser.parse_args(args)

    # Update variables with parsed arguments
    BEHAVIORS = args.behaviors
    NUM_CLASSES = len(BEHAVIORS)
    UMAP_DIM = args.umap_dim
    UMAP_NGH = args.umap_ngh
    UMAP_PATH = Path(args.umap_path)
    TEST_SET_PATH = args.test_set_path
    OUTPUT_PATH = args.output_path
    SAMPLE_SIZE = args.sample_size
    DATA_FORMAT = args.data_format
    SEED = args.seed

    BEHAVIOR_COLORS = [
        (0.9058823529411765, 0.5411764705882353, 0.7647058823529411),
        (0.9882352941176471, 0.5529411764705883, 0.3843137254901961),
        (0.5529411764705883, 0.6274509803921569, 0.796078431372549),
        (0.7019607843137254, 0.7019607843137254, 0.7019607843137254),
    ][:NUM_CLASSES]

    # Load data
    # ---------
    # Load test set
    rec_test, _, _ = load_records(DATA_FORMAT, TEST_SET_PATH)

    
    # Load data
    test_embeddings_2d = []
    test_labels = []

    for key, data in tqdm(rec_test):
        print(data.keys())
        test_labels.append(data["annotations"])
        test_embeddings_2d.append(
            pd.read_csv(
                UMAP_PATH / key / f"features_umap{UMAP_DIM}d{UMAP_NGH}_dimred.csv",
                index_col=0,
            ).values
        )

    test_labels = np.concatenate(test_labels)
    test_embeddings_2d = np.concatenate(test_embeddings_2d)

    # Plot UMAP
    # ---------
    fig, ax = plt.subplots()
    betp.plot_umap2d(
        data=test_embeddings_2d,
        labels=test_labels,
        sample_size=SAMPLE_SIZE,
        seed=SEED,
        cmap=mpl.colors.ListedColormap(BEHAVIOR_COLORS),
        cbar_label=None,
        cbar_ticklabels=BEHAVIORS,
        ax=ax,
    )

    fig.savefig(os.path.join(OUTPUT_PATH, "umap.png"))
    plt.close(fig)
    print('done')

if __name__ == "__main__":
    main()