import numpy as np
import matplotlib.pyplot as plt

X = np.array([[1, 1], [1, 2], [2, 1],
              [8, 8], [8, 9], [9, 8]])

np.random.seed(42)
W = np.random.rand(2, 2, 2)
lr = 0.5

for epoch in range(10):
    for x in X:
        d = np.linalg.norm(W - x, axis=2)
        bmu = np.unravel_index(np.argmin(d), d.shape)

        for i in range(2):
            for j in range(2):
                if abs(i-bmu[0]) <= 1 and abs(j-bmu[1]) <= 1:
                    W[i, j] += lr * (x - W[i, j])

    lr *= 0.9

test = np.array([8, 8])
d = np.linalg.norm(W - test, axis=2)
bmu = np.unravel_index(np.argmin(d), d.shape)
print("Winning neuron:", bmu)
print("Final weights:\n", W)

plt.scatter(X[:, 0], X[:, 1], label="Data")
for i in range(2):
    for j in range(2):
        plt.scatter(W[i, j, 0], W[i, j, 1], marker="^")
        plt.text(W[i, j, 0], W[i, j, 1], f"({i},{j})")

plt.title("Self-Organizing Map")
plt.legend()
plt.grid()
plt.show()