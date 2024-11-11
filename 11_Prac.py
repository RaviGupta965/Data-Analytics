import numpy as np

def k_means(X, k, max_iters=100):
    # Initialize centroids randomly
    centroids = X[np.random.choice(range(len(X)), k, replace=False)]

    for _ in range(max_iters):
        # Assign each data point to the nearest centroid
        labels = np.argmin(np.linalg.norm(X[:, np.newaxis] - centroids, axis=-1), axis=-1)

        # Update centroids
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])

        # Check convergence
        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    return labels, centroids

# Generate random data
np.random.seed(42)
X = np.random.rand(100, 2)

# Run k-means clustering
k = 4
labels, centroids = k_means(X, k)

print("Cluster labels:", labels)
print("Centroids:", centroids)
