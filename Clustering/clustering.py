import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from matplotlib import markers
import itertools


# finds indices of closest clusters to be merged on next iteration
# clusters_matrix - clusters matrix
# distance_col, cluster_col - ids of columns keeping min distance and closest cluster id
# returns cluster indices and distance between them
def find_clusters_to_merge(clusters_matrix, distance_col, cluster_col):
    min_dist = np.amin(clusters_matrix[:, distance_col], axis=0)
    index_1 = np.argmin(clusters_matrix[:, distance_col], axis=0)
    index_2 = int(clusters_matrix[index_1, cluster_col])
    return index_1, index_2, min_dist


# performs merge of clusters with indices c1_index, c2_index
# updates single-linkage distances in clusters_matrix
# updates cluster membership column in X_matrix
# c1_index, c2_index - indices of clusters to be merged
# X_matrix - data + cluster membership column
# distance_col, cluster_col - ids of columns keeping min distance and closest cluster id
# distances_matrix - initial pairwise distances matrix, good implementation doesn't need it in this method
def single_link_merge(c1_index, c2_index, X_matrix, clusters_matrix, distance_col, cluster_col, distances_matrix):
    for i in range(len(X_matrix)):
        if X_matrix[i, 2] == c2_index:
            X_matrix[i, 2] = c1_index

    for i in range(len(X_matrix)):
        if i != c1_index and i != c2_index:
            min_dist = np.minimum(clusters_matrix[c1_index, i], clusters_matrix[c2_index, i])
            clusters_matrix[c1_index, i] = min_dist
            clusters_matrix[i, c1_index] = min_dist

    for i in range(len(X_matrix)):
        clusters_matrix[c2_index, i] = np.inf
        clusters_matrix[i, c2_index] = np.inf


# performs merge of clusters with indices c1_index, c2_index
# updates complete-linkage distances in clusters_matrix
# updates cluster membership column in X_matrix
# c1_index, c2_index - indices of clusters to be merged
# X_matrix - data + cluster membership column
# distance_col, cluster_col - ids of columns keeping min distance and closest cluster id
# distances_matrix - initial pairwise distances matrix, good implementation doesn't need it in this method
def complete_link_merge(c1_index, c2_index, X_matrix, clusters_matrix, distance_col, cluster_col, distances_matrix):
    for i in range(len(X_matrix)):
        if X_matrix[i, 2] == c2_index:
            X_matrix[i, 2] = c1_index

    for i in range(len(X_matrix)):
        if i != c1_index and i != c2_index:
            max_dist = np.maximum(clusters_matrix[c1_index, i], clusters_matrix[c2_index, i])
            clusters_matrix[c1_index, i] = max_dist
            clusters_matrix[i, c1_index] = max_dist

    for i in range(len(X_matrix)):
        clusters_matrix[c2_index, i] = np.inf
        clusters_matrix[i, c2_index] = np.inf


# performs merge of clusters with indices c1_index, c2_index
# updates average-linkage distances in clusters_matrix
# updates cluster membership column in X_matrix
# c1_index, c2_index - indices of clusters to be merged
# X_matrix - data + cluster membership column
# distance_col, cluster_col - ids of columns keeping min distance and closest cluster id
# distances_matrix - initial pairwise distances matrix, use it for this method
def average_link_merge(c1_index, c2_index, X_matrix, clusters_matrix, distance_col, cluster_col, distances_matrix):
    elements_in_c1 = 0
    elements_in_c2 = 0

    for i in range(len(X_matrix)):
        if X_matrix[i, 2] == c1_index:
            elements_in_c1 += 1
        if X_matrix[i, 2] == c2_index:
            elements_in_c2 += 1
            X_matrix[i, 2] = c1_index

    for i in range(len(X_matrix)):
        if i != c1_index and i != c2_index:
            avg_dist = (clusters_matrix[c1_index, i] * elements_in_c1 + clusters_matrix[c2_index, i] * elements_in_c2) \
                       / (elements_in_c1 + elements_in_c2)
            clusters_matrix[c1_index, i] = avg_dist
            clusters_matrix[i, c1_index] = avg_dist

    for i in range(len(X_matrix)):
        clusters_matrix[c2_index, i] = np.inf
        clusters_matrix[i, c2_index] = np.inf


# the function which performs bottom-up (agglomerative) clustering
# merge_func - one of the three merge functions above, each with different linkage function
# X_matrix - data itself
# threshold - maximum merge distance, we stop merging if we reached it. if None, merge until there only is one cluster
def bottom_up_clustering(merge_func, X_matrix, distances_matrix, threshold=None):
    num_points = X_matrix.shape[0]

    # take dataset, add and initialize column for cluster membership
    X_data = np.c_[X_matrix, np.arange(0, num_points, 1)]

    # create clusters matrix, initially consisting of all points and pairwise distances
    # with last columns being distance to closest cluster and id of that cluster
    clusters = np.c_[distances_matrix, np.zeros((num_points, 2))]

    # ids of added columns - column with minimal distances, column with closest cluster ids
    dist_col_id = num_points
    clust_col_id = num_points + 1

    # calculate closest clusters and corresponding distances for each cluster
    clusters[:, clust_col_id] = np.argmin(clusters[:, :num_points], axis=1)
    clusters[:, dist_col_id] = np.amin(clusters[:, :num_points], axis=1)

    # array for keeping distances between clusters that we are merging
    merge_distances = np.zeros(num_points - 1)
    # main loop. at each step we are identifying and merging two closest clusters (wrt linkage function)
    for i in range(0, num_points - 1):
        c1_id, c2_id, distance = find_clusters_to_merge(clusters, dist_col_id, clust_col_id)
        # if threshold is set, we don't merge any further if we reached the desired max distance for merging
        if threshold is not None and distance > threshold:
            break
        merge_distances[i] = distance
        merge_func(c1_id, c2_id, X_data, clusters, dist_col_id, clust_col_id, distances_matrix)

        # calculate closest clusters and corresponding distances for each cluster
        clusters[:, clust_col_id] = np.argmin(clusters[:, :num_points], axis=1)
        clusters[:, dist_col_id] = np.amin(clusters[:, :num_points], axis=1)

        # uncomment when testing
        # print("Merging clusters #", c1_id, c2_id)
        # if i%30 == 0:
        #     for k, (marker, color) in zip(range(num_points), itertools.product(markers, colormap)):
        #         plt.scatter(X_data[X_data[:, 2] == k, 0], X_data[X_data[:, 2] == k, 1], color=color, marker=marker, label=k)
        #     plt.show()

    # todo use the plot below to find the optimal threshold to stop merging clusters
    plt.plot(np.arange(0, num_points - 1, 1), merge_distances[:num_points - 1])
    plt.title("Merge distances over iterations")
    plt.xlabel("Iteration #")
    plt.ylabel("Distance")
    plt.show()

    for k, (marker, color) in zip(range(num_points), itertools.product(markers, colormap)):
        plt.scatter(X_data[X_data[:, 2] == k, 0], X_data[X_data[:, 2] == k, 1], color=color, marker=marker)
    plt.title('Clusters of customers')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.show()


# importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values

# creating and populating matrix for storing pairwise distances
# diagonal elements are filled with np.inf to ease further processing
distances = squareform(pdist(X, metric='euclidean'))
np.fill_diagonal(distances, np.inf)

# seting up colors and marker types to use for plotting
markers = markers.MarkerStyle.markers
colormap = plt.cm.Dark2.colors

# performing bottom-up clustering with three different linkage functions
bottom_up_clustering(single_link_merge, X, distances, threshold=9.43)
bottom_up_clustering(complete_link_merge, X, distances, threshold=70)
bottom_up_clustering(average_link_merge, X, distances, threshold=41)
