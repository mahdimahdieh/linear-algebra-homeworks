import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import random
from PIL import Image


# Load the .mat file
def load_data(file_path):
    mat_contents = scipy.io.loadmat(file_path)
    data = mat_contents['Data']
    labels = mat_contents['labels']
    return data, labels


# Resize each image to the target size and flatten it
def preprocess_data(data, target_size=(28, 28)):
    flattened_data = []
    for i in range(len(data)):
        image = data[i][0]
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
            resized_image = pil_image.resize(target_size, Image.LANCZOS)
            flattened_image = np.array(resized_image).flatten()
            flattened_data.append(flattened_image)
    return np.array(flattened_data)


# Apply k-means clustering
def apply_kmeans(flattened_data, k_values):
    kmeans_models = {}
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(flattened_data)
        kmeans_models[k] = kmeans
    return kmeans_models


# Plot samples from each cluster
def plot_clusters(kmeans, flattened_data, k=5, samples_per_cluster=5, target_size=(28, 28)):
    labels = kmeans.labels_
    fig, axes = plt.subplots(k, samples_per_cluster, figsize=(10, 10))
    fig.suptitle(f'Cluster representatives for k={k}')

    for cluster_id in range(k):
        cluster_indices = np.where(labels == cluster_id)[0]
        random_indices = random.sample(list(cluster_indices), samples_per_cluster)
        for i, idx in enumerate(random_indices):
            axes[cluster_id, i].imshow(flattened_data[idx].reshape(target_size), cmap='gray')
            axes[cluster_id, i].axis('off')

    plt.tight_layout()
    plt.show()


# Main function to execute the entire workflow
def main():
    file_path = 'Data_hoda_full.mat'  # Update with your local file path
    data, labels = load_data(file_path)

    # Preprocess data
    target_size = (28, 28)
    flattened_data = preprocess_data(data, target_size)

    # Apply k-means clustering
    k_values = [3, 4, 5, 6, 7]
    kmeans_models = apply_kmeans(flattened_data, k_values)

    # Plot clusters for k = 5
    k = 7
    plot_clusters(kmeans_models[k], flattened_data, k=k, target_size=target_size)


if __name__ == "__main__":
    main()
