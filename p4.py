import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

X, y = make_blobs(n_samples=500, centers=3, random_state=23)

X = StandardScaler().fit_transform(X)

k = 3
np.random.seed(23)
centers = X[np.random.choice(len(X), k, replace=False)]

for _ in range(100):
    labels = np.argmin(
        np.linalg.norm(X[:, None] - centers, axis=2), axis=1
    )
    new_centers = np.array([
        X[labels == i].mean(axis=0) if np.any(labels == i)
        else centers[i] for i in range(k)
    ])

    if np.allclose(centers, new_centers):
        break
    centers = new_centers

plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.scatter(centers[:, 0], centers[:, 1], c="red", marker="^")
plt.title("K-Means Clustering")
plt.grid(True)
plt.show()