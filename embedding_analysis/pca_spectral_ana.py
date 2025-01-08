import numpy as np
from cuml.cluster import KMeans
from cuml import PCA
from sklearn.metrics import silhouette_score
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt
import pandas as pd
import os
import json
from tqdm import tqdm
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)-8s : %(message)s')

def load_embedding(datapath):
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

    return df

# Step 2: K-Means Clustering and Finding the Optimal Number of Clusters
def kmeans_clustering_mini_batch(data_train, data_eval, max_clusters=10, min_clusters=2, step=1, max_iter=1000, batch_size=2048):
    inertia = []
    silhouette_tr = []
    silhouette_te = []
    logging.info("Running KMeans clustering")
    for k in tqdm(range(min_clusters, max_clusters + 1, step)):
        kmeans = KMeans(n_clusters=k, n_init=10, max_iter=max_iter, batch_size=batch_size)
        labels_tr = kmeans.fit_predict(data_train)
        labels_te = kmeans.predict(data_eval)
        inertia.append(kmeans.inertia_)

        silhouette_tr.append(silhouette_score(data_train, labels_tr))
        silhouette_te.append(silhouette_score(data_eval, labels_te))
    return inertia, silhouette_tr, silhouette_te


def main(args=None):

    argu = argument_parser(args)

    if argu.output is None:
        argu.output = os.path.join(argu.input, "PCA_output")
    os.makedirs(argu.output, exist_ok=True)

    datapath = os.path.join(argu.input, "embedding_train.numpy")
    dataval = os.path.join(argu.input, "embedding_test.numpy")

    # Load your temporal encoded data (shape: n_samples, 128 dimensions)
    train_data =  load_embedding(datapath)# Replace with your data
    tr = train_data.drop(columns='video').to_numpy()

    eval_data = load_embedding(dataval)
    te = eval_data.drop(columns='video').to_numpy()

    if argu.debug:
        tr = tr[:100]
        te = te[:100] 
        argu.max_clusters = 10
        argu.min_clusters = 5
        argu.step = 5
        argu.max_iter = 10
        argu.batch_size = 4096


    with open(os.path.join(argu.output, "args.txt"), "w") as f:
       json.dump(argu.__dict__, f, indent=4)
       
    # Step 1: Dimensionality Reduction with PCA
    logging.info("Running PCA")
    reducer = PCA(n_components=2)

    data_train = reducer.fit_transform(tr)
    data_eval = reducer.transform(te)


    # Run clustering
    inertia, silhouette_tr, silhouette_te = kmeans_clustering_mini_batch(data_train, 
                                                                         data_eval, 
                                                                         argu.max_clusters, 
                                                                         argu.min_clusters, 
                                                                         argu.step,
                                                                         argu.max_iter,
                                                                         argu.batch_size)

    # Step 3: Plot Inertia and Silhouette Scores
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    ax[0].plot(range(argu.min_clusters, argu.max_clusters + 1, argu.step), inertia, marker='o')
    ax[0].set_title("Elbow Method (Inertia)")
    ax[0].set_xlabel("Number of Clusters")
    ax[0].set_ylabel("Inertia")

    ax[1].plot(range(argu.min_clusters, argu.max_clusters + 1, argu.step), silhouette_tr, marker='o')
    ax[1].set_title("Silhouette Score - training data")
    ax[1].set_xlabel("Number of Clusters")
    ax[1].set_ylabel("Silhouette Score")

    ax[2].plot(range(argu.min_clusters, argu.max_clusters + 1, argu.step), silhouette_te, marker='o')
    ax[2].set_title("Silhouette Score - eval data")
    ax[2].set_xlabel("Number of Clusters")
    ax[2].set_ylabel("Silhouette Score")
    fig.savefig(os.path.join(argu.output, "elbow_silhouette.png"), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    logging.info("Elbow and Silhouette plots saved to disk")    

    # Step 4: Visualization of Clusters
    logging.info("Visualizing Clusters")
    optimal_clusters = silhouette_tr.index(max(silhouette_tr)) + 2
    kmeans = KMeans(n_clusters=optimal_clusters, n_init=10, max_iter=argu.max_iter, batch_size=argu.batch_size)
    labels_tr = kmeans.fit_predict(data_train)
    labels_te = kmeans.predict(data_eval)

    logging.info(f"Optimal number of clusters: {optimal_clusters}")

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].scatter(data_train[:, 0], data_train[:, 1], c=labels_tr, cmap='viridis', s=10)
    ax[0].set_title(f"training data")
    ax[0].set_xlabel("PCA Component 1")
    ax[0].set_ylabel("PCA Component 2")

    ax[1].scatter(data_eval[:, 0], data_eval[:, 1], c=labels_te, cmap='viridis', s=10)
    ax[1].set_title( "eval data")
    ax[1].set_xlabel("PCA Component 1")
    ax[1].set_ylabel("PCA Component 2")

    fig.suptitle(f"K-Means Clustering on PCA with {optimal_clusters} Clusters")
    fig.savefig(os.path.join(argu.output, "PCA_clusters.png"), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    
    logging.info("Into spectral study.")

    cov_matrix = np.cov(tr.T)  # Transpose data to have features as rows
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # plot eigenvalues
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(eigenvalues[::-1])
    ax.set_title("Eigenvalues")
    ax.set_xlabel("Index")
    ax.set_ylabel("Eigenvalue")
    fig.savefig(os.path.join(argu.output, "eigenvalues.png"), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    top_k = 5  # Adjust based on the elbow point in eigenvalues plot
    spectral = SpectralClustering(n_clusters=top_k, affinity='nearest_neighbors', random_state=42)
    labels_tr = spectral.fit_predict(tr)
    labels_te = spectral.fit_predict(te)

    silhouette_tr = silhouette_score(tr, labels_tr)
    silhouette_te = silhouette_score(te, labels_te)
    logging.info(f"Silhouette Score - training data: {silhouette_tr}")
    logging.info(f"Silhouette Score - eval data: {silhouette_te}")

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].scatter(data_train[:, 0], data_train[:, 1], c=labels_tr, cmap='viridis', s=10)
    ax[0].set_title(f"training data")
    ax[0].set_xlabel("PCA Component 1")
    ax[0].set_ylabel("PCA Component 2")

    ax[1].scatter(data_eval[:, 0], data_eval[:, 1], c=labels_te, cmap='viridis', s=10)
    ax[1].set_title( "eval data")
    ax[1].set_xlabel("PCA Component 1")
    ax[1].set_ylabel("PCA Component 2")

    fig.suptitle(f"Spectral Clustering on PCA with {top_k} Clusters")
    fig.savefig(os.path.join(argu.output, "spectral_clusters.png"), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    logging.info("Spectral Clustering completed")

    return

def argument_parser(args=None):
    parser = argparse.ArgumentParser(description='PCA & spectral')
    parser.add_argument('--input', type=str, help='path to the directory containing the embeddings', required=True)
    parser.add_argument('--output', type=str, help='path to the output directory')    
    parser.add_argument('--min_clusters', type=int, help='minimum number of clusters', default=5)
    parser.add_argument('--max_clusters', type=int, help='maximum number of clusters', default=100)
    parser.add_argument('--step', type=int, help='step size for number of clusters', default=5)
    parser.add_argument('--batch_size', type=int, help='batch size for mini-batch KMeans', default=4096)
    parser.add_argument('--max_iter', type=int, help='maximum number of iterations for KMeans', default=1000)
    parser.add_argument('--debug', action='store_true', help='enable debug mode')
    arguments = parser.parse_args(args)
    return arguments


if __name__ == "__main__":
    # argument = ["--input", r"C:\Users\chataint\Documents\projet\humanlisbet\results\bet_embedders\bet_embedders\13879972", 
    #             "--output", r"C:\Users\chataint\Documents\projet\humanlisbet\test",
    #             "--min_clusters", "5",
    #             "--max_clusters", "10",
    #             "--step", "5",
    #             "--batch_size", "4096",
    #             "--max_iter", "10",
    #             "--debug"]
    # main(argument) 
    main()