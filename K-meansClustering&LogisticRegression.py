import numpy as np
import matplotlib.pyplot as plt

# Step 1: Input data
points = np.array([
    [185, 72],
    [170, 54],
    [168, 60],
    [179, 68],
    [183, 72],
    [188, 77]
])

# Step 2: Initialize centroids manually
centroids = np.array([
    [170, 54],   # Centroid 0
    [183, 72]    # Centroid 1
])

def euclidean(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))

# Step 3: Loop until convergence
iteration = 0
while True:
    print(f"\nIteration {iteration + 1}")
    # Step 4: Assign each point to nearest centroid
    clusters = {0: [], 1: []}
    for point in points:
        distances = [euclidean(point, centroids[i]) for i in range(2)]
        cluster_index = np.argmin(distances)
        clusters[cluster_index].append(point)

    # Step 5: Save current centroids to check convergence
    old_centroids = centroids.copy()

    # Step 6: Recompute centroids
    for i in range(2):
        if clusters[i]:
            centroids[i] = np.mean(clusters[i], axis=0)

    print("Updated Centroids:\n", centroids)

    # Step 7: Check for convergence (if centroids don’t change)
    if np.allclose(centroids, old_centroids):
        print("\n✅ Converged!")
        break

    iteration += 1

# Step 8: Final cluster results
for i in range(2):
    print(f"\nCluster {i}:")
    for point in clusters[i]:
        print(point)

# Optional: Visualize
colors = ['red', 'blue']
for i in range(2):
    cluster_points = np.array(clusters[i])
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors[i], label=f'Cluster {i}')
    plt.scatter(centroids[i][0], centroids[i][1], color='black', marker='x', s=100)

plt.xlabel("Height")
plt.ylabel("Weight")
plt.title("Manual K-Means Clustering")
plt.grid(True)
plt.legend()
plt.show()
