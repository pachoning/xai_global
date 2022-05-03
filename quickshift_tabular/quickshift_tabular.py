import numpy as np
from skimage.segmentation import find_boundaries

def get_neighbors(mask, element):
    
    unique_superpixels = np.unique(mask)
    total_sp = len(unique_superpixels)
    neighbors = []
    element_mask = mask == element
    for candiate in unique_superpixels:
        if candiate != element:
            candidate_mask = mask == candiate
            candidate_boundary = find_boundaries(candidate_mask, mode='thick')
            product = np.multiply(element_mask, candidate_boundary)
            is_neighbor = np.any(product)
            if is_neighbor:
                neighbors.append(candiate)
    return neighbors

def quickshift_tab(X, superpixels_order, mask,
                   ratio=1.0, kernel_size=5, max_dist=10,
                   random_seed=42):

    if kernel_size < 1:
        raise ValueError("`kernel_size` should be >= 1.")
        
    X = ratio * X

    random_state = np.random.default_rng(random_seed)

    # window size for neighboring pixels to consider
    inv_kernel_size_sqr = -0.5 / (kernel_size * kernel_size)

    n = X.shape[0]
    p = X.shape[1]

    densities = np.zeros(n)
    densities += random_state.normal(scale=0.00001, size=n)

    parents = np.copy(superpixels_order)
    dist_parents = np.zeros(n)

    # compute densities
    for i in range(n):
        cum_feat_dist = 0
        current_point = X[i,:]
        for i_ in range(n):
            other_point = X[i_,:]
            feat_dist = np.sum(np.square(current_point - other_point))
            cum_feat_dist += feat_dist
        cum_feat_dist = inv_kernel_size_sqr*cum_feat_dist
        densities[i] += np.exp(cum_feat_dist)

    # find nearest node with higher density
    for i in range(n):
        # Take information for the current superpixel
        current_superpixel = superpixels_order[i]
        current_point = X[i,:]
        current_densisty = densities[i]
        
        # Take all the neighbors
        neighbors = get_neighbors(mask, current_superpixel)
        
        # If there are neighbors, get all with higher density
        total_neighbors = len(neighbors)
        min_dist = np.inf
        if  total_neighbors > 0:
            for j in range(total_neighbors):
                neighbor = neighbors[j]
                idx_neighbor = np.where(superpixels_order == neighbor)
                idx_neighbor = idx_neighbor[0][0]
                neighbor_density = densities[idx_neighbor]
                if neighbor_density > current_densisty:
                    neighbor_point = X[idx_neighbor,:]
                    dist_features = np.sum(np.square(current_point - neighbor_point))
                    # Get the closest neighbors (in terms of feature space distance)
                    if dist_features < min_dist:
                        min_dist = dist_features
                        parents[i] = neighbor
                        dist_parents[i] = dist_features
    
    # Do not join if distance is greater than the threshold
    for i in range(n):
        current_dist_parents = dist_parents[i]
        if current_dist_parents > max_dist:
            parents[i] = superpixels_order[i]
            dist_parents[i] = 0

    # Build the tree
    original_parent = np.empty(shape=superpixels_order.shape, dtype=superpixels_order.dtype)
    for i in range(n):
        current_superpixel = superpixels_order[i]
        parent = parents[i]
        old = None
        while (old is None) or (old is not None and old != parent):
            old = parent
            idx_parent = np.where(superpixels_order == parent)
            idx_parent = idx_parent[0][0]
            parent = parents[idx_parent]
        original_parent[i] = parent
    return original_parent
