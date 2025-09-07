# K-means Clustering Project

---
This project demonstrates how I implemented the **K-means algorithm** from scratch in Python and applied it to two main tasks:

1. Understanding clustering behavior on a toy dataset.
2. Using clustering for **image compression** by reducing the number of unique colors in an image.

Through this, I show how K-means works step-by-step, including centroid initialization, iterative refinement, and how compression can drastically reduce image size while maintaining visual quality.

---

## Outline

* [1 - Implementing K-means](#1---implementing-k-means)

  * [1.1 Finding closest centroids](#11-finding-closest-centroids)
  * [1.2 Computing centroid means](#12-computing-centroid-means)
* [2 - K-means on a sample dataset](#2---k-means-on-a-sample-dataset)
* [3 - Random initialization](#3---random-initialization)
* [4 - Image compression with K-means](#4---image-compression-with-k-means)

  * [4.1 Dataset](#41-dataset)
  * [4.2 K-means on image pixels](#42-k-means-on-image-pixels)
  * [4.3 Compressing the image](#43-compressing-the-image)

---

<a name="1---implementing-k-means"></a>

## 1 - Implementing K-means

The K-means algorithm groups data points into **K clusters** by iteratively:

1. Assigning each point to its closest centroid.
2. Recomputing centroids as the mean of assigned points.

Here’s the pseudocode:

```python
centroids = kMeans_init_centroids(X, K)

for iter in range(iterations):
    idx = find_closest_centroids(X, centroids)
    centroids = compute_centroids(X, idx, K)
```

---

<a name="11-finding-closest-centroids"></a>

### 1.1 Finding closest centroids

The first step is assigning each training example to its closest centroid.

✅ **My implementation:**

```python
import numpy as np

def find_closest_centroids(X, centroids):
    """
    Computes the centroid memberships for every example.
    
    Args:
        X (ndarray): (m, n) Input values      
        centroids (ndarray): (K, n) centroids
    
    Returns:
        idx (array_like): (m,) closest centroids
    """
    K = centroids.shape[0]
    idx = np.zeros(X.shape[0], dtype=int)

    for i in range(X.shape[0]):
        distances = []
        for j in range(K):
            norm_ij = np.linalg.norm(X[i] - centroids[j])
            distances.append(norm_ij)
        idx[i] = np.argmin(distances)
    
    return idx
```

---

<a name="12-computing-centroid-means"></a>

### 1.2 Computing centroid means

The second step is recomputing each centroid as the average of all points assigned to it.

✅ **My implementation:**

```python
def compute_centroids(X, idx, K):
    """
    Returns the new centroids by computing the means of 
    the data points assigned to each centroid.
    
    Args:
        X (ndarray):   (m, n) Data points
        idx (ndarray): (m,) Index of closest centroid for each example
        K (int):       Number of centroids
    
    Returns:
        centroids (ndarray): (K, n) New centroids computed
    """
    m, n = X.shape
    centroids = np.zeros((K, n))

    for k in range(K):
        points = X[idx == k]
        if len(points) > 0:
            centroids[k] = np.mean(points, axis=0)
    
    return centroids
```

---

<a name="2---k-means-on-a-sample-dataset"></a>

## 2 - K-means on a sample dataset

I ran K-means on a toy dataset to visualize the clustering process. Each iteration shows centroids moving closer to their optimal positions. The final centroids are marked as black "X"s in the plot.

---

<a name="3---random-initialization"></a>

## 3 - Random initialization

To avoid poor clustering due to bad initial centroids, I implemented **random initialization**:

```python
def kMeans_init_centroids(X, K):
    randidx = np.random.permutation(X.shape[0])
    centroids = X[randidx[:K]]
    return centroids
```

Running K-means multiple times with different random seeds produces different clustering outcomes.

---

<a name="4---image-compression-with-k-means"></a>

## 4 - Image compression with K-means

Finally, I applied K-means to image compression.

* Every pixel is treated as a 3D point in RGB space.
* By clustering pixels into 16 groups, I reduced the color palette to just **16 colors**.
* Each pixel is then replaced with its closest centroid color.

This reduces memory usage from **393,216 bits → 65,920 bits**, about a **6x compression**.

---

<a name="41-dataset"></a>

### 4.1 Dataset

I used `bird_small.png` as the input image. Each pixel was reshaped into a 2D matrix `X_img` with shape `(16384, 3)`.

---

<a name="42-k-means-on-image-pixels"></a>

### 4.2 K-means on image pixels

After running K-means with `K=16`, the 16 representative colors were extracted as centroids.

---

<a name="43-compressing-the-image"></a>

### 4.3 Compressing the image

Each pixel was reassigned to its centroid color, and the compressed image was reconstructed. The result retained most details while using only 16 colors.

| Original Image                     | Compressed Image                     |
| ---------------------------------- | ------------------------------------ |
| ![original](images/figure%202.png) | ![compressed](images/figure%203.png) |

---

## Key Takeaways

* K-means clustering can be implemented **from scratch** with just NumPy.
* Random initialization is crucial for better clustering results.
* K-means has practical applications like **image compression**, where storage is significantly reduced with minimal quality loss.

---

## Next Steps

In future work, I plan to extend this by:

* Trying different values of `K` for image compression.
* Comparing K-means results with more advanced clustering algorithms like **Gaussian Mixture Models**.
* Applying K-means to other real-world datasets.

---
