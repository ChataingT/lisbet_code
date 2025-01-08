import numpy as np
import umap
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score
<<<<<<< Updated upstream
=======

>>>>>>> Stashed changes
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd
import os
import json
from tqdm import tqdm
import argparse
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)-8s : %(message)s')

def load_embedding(datapath, label_path):
    
    emb_train = np.load(datapath, allow_pickle=True)

    # Initialize an empty list to collect rows for the DataFrame
    rows = []

    # Iterate over the array to flatten the structure
    for video, coords in emb_train:
        for idx, coord in enumerate(coords):
            rows.append([video, idx] + coord.tolist())

    # Create a DataFrame
    df = pd.DataFrame(rows, columns=['video', 'frame'] + [f'em_{i}' for i in range(len(coord))])
    df.video = df.video.astype(np.int64)

    df = df.drop(columns='frame')

    logging.info(f"Data loaded from {datapath}")

    lb = pd.read_excel(label_path)[['VCFS_DATABASE_ADMIN 2::Sujet_ID', 'VCFS_DATABASE_ADMIN 2::Diagnosis']]
    lb.rename(columns={'VCFS_DATABASE_ADMIN 2::Sujet_ID':'video', 'VCFS_DATABASE_ADMIN 2::Diagnosis':'diag'}, inplace=True)
    
    lb['diag'] = lb['diag'].replace({'Low-Risk':'TD'})
    lb['diag'] = lb['diag'].replace({'Normal_Control':'TD'})
    lb['diag'] = lb['diag'].replace({'Autism':'ASD'})
    mapping = {"ASD":0, "TD":1}
    lb.diag =lb.diag.map(mapping)
    df = pd.merge(left=df, right=lb, how='left', on='video')
    labels = df.diag
    df = df.drop(columns='diag')
    return df, labels

# Step 2: K-Means Clustering and Finding the Optimal Number of Clusters
def kmeans_clustering_mini_batch(data_eval, max_clusters=10, min_clusters=2, step=1, max_iter=1000, batch_size=2048):
    inertia = []
    silhouette_te = []
    logging.info("Running KMeans clustering")
    for k in tqdm(range(min_clusters, max_clusters + 1, step)):
        kmeans = MiniBatchKMeans(n_clusters=k, n_init=10, max_iter=max_iter, batch_size=batch_size)
        labels_te = kmeans.fit_predict(data_eval)
        inertia.append(kmeans.inertia_)

        silhouette_te.append(silhouette_score(data_eval, labels_te))
    return inertia, silhouette_te


def main(args=None):

    argu = argument_parser(args)

    if argu.output is None:
        argu.output = os.path.join(argu.input, "umap_output")
    os.makedirs(argu.output, exist_ok=True)


    # datapath = os.path.join(argu.input, "embedding_train.numpy")
    dataval = os.path.join(argu.input, "embedding_test.numpy")

    # Load your temporal encoded data (shape: n_samples, 128 dimensions)

    eval_data, labels = load_embedding(dataval, argu.label)
    te = eval_data.drop(columns='video').to_numpy()

    if argu.debug:
        te = te[:100] 
        argu.max_clusters = 10
        argu.min_clusters = 5
        argu.step = 5
        argu.max_iter = 10
        argu.batch_size = 4096

    print(te)
    return
    with open(os.path.join(argu.output, "args.txt"), "w") as f:
       json.dump(argu.__dict__, f, indent=4)
       
    # Step 1: Dimensionality Reduction with UMAP
    logging.info("Running UMAP")
    umap_reducer = Pipeline(
<<<<<<< Updated upstream
        [
            ("scaler", MinMaxScaler()),
            ("reducer", umap.UMAP(n_neighbors=60, verbose=True)),
        ]
=======
    [
        ("scaler", MinMaxScaler()),
        ("reducer", umap.UMAP(n_neighbors=60, verbose=True)),
    ]
>>>>>>> Stashed changes
    )

    data_umap_eval = umap_reducer.fit_transform(te)


    # Plot UMAP
    fig, ax = plt.subplots(figsize=(2.2, 2.5))

    # Sample 10K points
    if len(data_umap_eval) < 10000:
        sample_idx = [i for i in range(len(data_umap_eval))]
    else:
        rng = np.random.default_rng(1789)
        sample_idx = rng.choice(data_umap_eval.shape[0], replace=False, size=10000)

    # Scale marker size to compensate for unbalanced classes
    # marker_size = 0.5 * np.array([class_weight[label] for label in test_labels])

    # Plot data
    sc = ax.scatter(
        data_umap_eval[sample_idx, 0],
        data_umap_eval[sample_idx, 1],
        # s=marker_size[sample_idx],
        c=labels[sample_idx],
        marker=".",
        # vmin=-0.5,
        # vmax=NUM_CLASSES_TASK1 - 0.5,
        # cmap=plotting.get_custom_cmap(NUM_CLASSES_TASK1),
    )

    fig.savefig(os.path.join(argu.output, "umap.png"), dpi=300, bbox_inches='tight', facecolor='white')

    # Run clustering
    inertia, silhouette_te = kmeans_clustering_mini_batch(data_umap_eval, 
<<<<<<< Updated upstream
                                                        argu.max_clusters, 
                                                        argu.min_clusters, 
                                                        argu.step,
                                                        argu.max_iter,
                                                        argu.batch_size)
=======
                                                                         argu.max_clusters, 
                                                                         argu.min_clusters, 
                                                                         argu.step,
                                                                         argu.max_iter,
                                                                         argu.batch_size)
>>>>>>> Stashed changes

    # Step 3: Plot Inertia and Silhouette Scores
    fig, ax = plt.subplots(1, 2, figsize=(17, 5))
    ax[0].plot(range(argu.min_clusters, argu.max_clusters + 1, argu.step), inertia, marker='o')
    ax[0].set_title("Elbow Method (Inertia)")
    ax[0].set_xlabel("Number of Clusters")
    ax[0].set_ylabel("Inertia")

<<<<<<< Updated upstream

=======
>>>>>>> Stashed changes
    ax[1].plot(range(argu.min_clusters, argu.max_clusters + 1, argu.step), silhouette_te, marker='o')
    ax[1].set_title("Silhouette Score - eval data")
    ax[1].set_xlabel("Number of Clusters")
    ax[1].set_ylabel("Silhouette Score")
    fig.savefig(os.path.join(argu.output, "elbow_silhouette.png"), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    logging.info("Elbow and Silhouette plots saved to disk")    

    # Step 4: Visualization of Clusters
    logging.info("Visualizing Clusters")
    optimal_clusters = silhouette_te.index(max(silhouette_te)) + 2
    kmeans = MiniBatchKMeans(n_clusters=optimal_clusters, n_init=10, max_iter=argu.max_iter, batch_size=argu.batch_size)
    labels_te = kmeans.fit_predict(data_umap_eval)

    logging.info(f"Optimal number of clusters: {optimal_clusters}")

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    ax.scatter(data_umap_eval[:, 0], data_umap_eval[:, 1], c=labels_te, cmap='viridis', s=10)
    ax.set_title( "eval data")
    ax.set_xlabel("UMAP Component 1")
    ax.set_ylabel("UMAP Component 2")

    fig.suptitle(f"K-Means Clustering on UMAP with {optimal_clusters} Clusters")
    fig.savefig(os.path.join(argu.output, "umap_clusters.png"), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    logging.info("Done.")
    return

def argument_parser(args=None):
    parser = argparse.ArgumentParser(description='UMAP')
    parser.add_argument('--input', type=str, help='path to the directory containing the embeddings', required=True)
    parser.add_argument('--output', type=str, help='path to the output directory')    
    parser.add_argument('--label', type=str, help='path to the label file')    
    parser.add_argument('--min_clusters', type=int, help='minimum number of clusters', default=5)
    parser.add_argument('--max_clusters', type=int, help='maximum number of clusters', default=100)
    parser.add_argument('--step', type=int, help='step size for number of clusters', default=5)
    parser.add_argument('--batch_size', type=int, help='batch size for mini-batch KMeans', default=4096)
    parser.add_argument('--max_iter', type=int, help='maximum number of iterations for KMeans', default=1000)
    parser.add_argument('--debug', action='store_true', help='enable debug mode')
    arguments = parser.parse_args(args)
    return arguments


if __name__ == "__main__":
<<<<<<< Updated upstream
    argument = ["--input", r"C:\Users\chataint\Documents\projet\humanlisbet\results\bet_embedders\bet_embedders\13879972", 
                "--output", r"C:\Users\chataint\Documents\projet\humanlisbet\test",
                "--min_clusters", "5",
                "--max_clusters", "100",
                "--step", "5",
                "--batch_size", "8112",
=======
    # argument = ["--input", r"C:\Users\chataint\Documents\projet\humanlisbet\results\bet_embedders\bet_embedders\13879972", 
    #             "--output", r"C:\Users\chataint\Documents\projet\humanlisbet\test",
    #             "--min_clusters", "5",
    #             "--max_clusters", "10",
    #             "--step", "5",
    #             "--batch_size", "4096",
    #             "--max_iter", "10",
    #             "--debug"]
    # main(argument) 
    argument = ["--input", r"/home/share/schaer2/thibaut/humanlisbet/bet_embedders/13879972", 
                "--output", r"/home/share/schaer2/thibaut/humanlisbet/test",
                "--label", r"/home/share/schaer2/thibaut/humanlisbet/datasets/humans/data_mapping.xlsx",
                "--min_clusters", "5",
                "--max_clusters", "100",
                "--step", "5",
                "--batch_size", "24444",
>>>>>>> Stashed changes
                "--max_iter", "1000",
                # "--debug"
                ]
    main(argument) 
    # main()