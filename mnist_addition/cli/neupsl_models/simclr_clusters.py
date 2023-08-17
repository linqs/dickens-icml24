import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sys
import torch
import torchvision

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

THIS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.join(THIS_DIR, '..', '..', '..'))

from mnist_addition.cli.neupsl_models.mlp import MLP


def main():
    # Load the simclr pretrained model.
    resnet18_ = torchvision.models.resnet18()
    resnet18_.conv1 = torch.nn.Conv2d(1, resnet18_.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
    backbone = torchvision.models.resnet18(num_classes=256)

    data_folder = f"{THIS_DIR}/../../data/experiment::mnist-1/split::0/train-size::6000/overlap::0.00"

    backbone.load_state_dict(torch.load(
        f"{data_folder}/saved-networks/simclr-pretrained-backbone.pt"
    ))

    projection_head = MLP(256, 256, 256, 2)
    projection_head.load_state_dict(torch.load(
        f"{data_folder}/saved-networks/simclr-pretrained-projection-head.pt"
    ))

    # Load the mnist dataset.
    training_indices = torch.tensor(
        pd.read_csv(f"{data_folder}/image-target-train.txt", sep="\t", header=None)[0].unique())
    all_features = pd.read_csv(f"{data_folder}/entity-data-map.txt", sep="\t", header=None, index_col=0)
    training_features = torch.tensor(all_features.loc[training_indices].iloc[:, :-1].values, dtype=torch.float32)
    training_labels = all_features.loc[training_indices].iloc[:, -1].values
    training_features = training_features.reshape(-1, 1, 28, 28)
    training_features = training_features.expand(-1, 3, -1, -1)

    # Get the features from the backbone.
    features = backbone(training_features)

    # Dimensionality reduction.
    dim_reduced_features = PCA(n_components=10).fit_transform(features.detach().numpy())

    # Cluster the features.
    kmeans_clustering = KMeans(n_clusters=10, n_init='auto').fit_predict(dim_reduced_features)

    # Plot the clusters.
    two_d_features = TSNE(n_components=2).fit_transform(dim_reduced_features)

    fig, ax = plt.subplots()
    for i in range(10):
        ax.scatter(two_d_features[training_labels == i, 0], two_d_features[training_labels == i, 1], label=i)
    ax.legend()
    ax.grid(True)
    plt.show()

    pd.DataFrame(np.stack([training_labels, kmeans_clustering], axis=1), columns=["label", "cluster"]).to_csv(
        f"{data_folder}/simclr-clusters.txt", sep="\t", index=False)


if __name__ == "__main__":
    main()
