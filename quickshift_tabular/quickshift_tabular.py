import numpy as np
import matplotlib.pyplot as plt 

def quickshift_tab(X, ratio=1.0, kernel_size=5, max_dist=10,
                   return_tree=False, random_seed=42):
    """Segments image using quickshift clustering in Color-(x,y) space.

    Produces an oversegmentation of the image using the quickshift mode-seeking
    algorithm.

    Parameters
    ----------
    X : (n,2+p) np.array
        Input data matrix. Columns 0 and 1 are the spatial coordinates
    ratio : float, optional, between 0 and 1
        Balances features proximity and space proximity.
        Higher values give more weight to features.
    kernel_size : float, optional
        Width of Gaussian kernel used in smoothing the
        sample density. Higher means fewer clusters.
    max_dist : float, optional
        Cut-off point for data distances.
        Higher means fewer clusters.
    return_tree : bool, optional
        Whether to return the full segmentation hierarchy tree and distances.
    random_seed : int, optional
        Random seed used for breaking ties.

    Returns
    -------
    cluster_list : (n,1) ndarray
        Integer array indicating cluster labels.

    References
    ----------
    .. [1] Quick shift and kernel methods for mode seeking,
           Vedaldi, A. and Soatto, S.
           European Conference on Computer Vision, 2008
    """

    if kernel_size < 1:
        raise ValueError("`kernel_size` should be >= 1.")

    #image = gaussian(image, [sigma, sigma, 0], mode='reflect', channel_axis=-1)
    #image = np.ascontiguousarray(image * ratio)
    X[:,2:] = X[:,2:] * ratio

    cluster_list = quickshift_tab_core(
        X, kernel_size=kernel_size, max_dist=max_dist,
        return_tree=return_tree, random_seed=random_seed)
    return cluster_list

def quickshift_tab_core(X, kernel_size,max_dist, return_tree, random_seed):
    """Clusters a data matrix using quickshift clustering in 
    (x,y)-features space.

    Produces a clustering of the data matrix using the quickshift mode-seeking
    algorithm.

    Parameters
    ----------
    X : (x, y, feature_1,...,feature_p) np.array
        Input image.
    kernel_size : float
        Width of Gaussian kernel used in smoothing the
        sample density. Higher means fewer clusters.
    max_dist : float
        Cut-off point for data distances.
        Higher means fewer clusters.
    return_tree : bool
        Whether to return the full segmentation hierarchy tree and distances.
    random_seed : {None, int, `numpy.random.Generator`}, optional
        If `random_seed` is None the `numpy.random.Generator` singleton
        is used.
        If `random_seed` is an int, a new ``Generator`` instance is used,
        seeded with `random_seed`.
        If `random_seed` is already a ``Generator`` instance then that instance
        is used.

        Random seed used for breaking ties.

    Returns
    -------
    cluster_list : (n,1) np.array
        Integer mask indicating segment labels.
    """

    random_state = np.random.default_rng(random_seed)


    # TODO join orphaned roots?
    # Some nodes might not have a point of higher density within the
    # search window. We could do a global search over these in the end.
    # Reference implementation doesn't do that, though, and it only has
    # an effect for very high max_dist.

    # window size for neighboring pixels to consider
    inv_kernel_size_sqr = -0.5 / (kernel_size * kernel_size)
    kernel_width = np.ceil(3 * kernel_size)

    # cdef Py_ssize_t height = image.shape[0]
    # cdef Py_ssize_t width = image.shape[1]
    # cdef Py_ssize_t channels = image.shape[2]
    
    n = X.shape[0]
    p = X.shape[1] - 2

    densities = np.zeros(n)
    current_point = np.zeros(p+2)

    #cdef np_floats current_density, closest, dist, t
    #cdef Py_ssize_t i, j, r_min, r_max, c_min, c_max
    #closest = np.inf

    # this will break ties that otherwise would give us headache
    densities += random_state.normal(scale=0.00001, size=n)

    # default parent to self
    parent = np.arange(n)
    dist_parent = np.zeros(n)

    # compute densities
    #with nogil:
    for i in range(n):
        current_point = X[i,:]
        for i_ in range(n):
            dist = 0
            for j in range(p+2):
                t = (current_point[j]-X[i_, j])
                dist += t * t
                densities[i] += np.exp(dist * inv_kernel_size_sqr)

    # find nearest node with higher density
    for i in range(n):
        current_point = X[i,:]
        closest = np.inf
        current_density = densities[i]
        for i_ in range(n):
            if densities[i_] > current_density:
                dist = 0
                # We compute the distances twice since otherwise
                # we get crazy memory overhead
                # (width * height * windowsize**2)
                for j in range(p+2):
                    t = (current_point[j]-X[i_, j])
                    dist += t * t
                if dist < closest:
                    closest = dist
                    parent[i] = i_
        dist_parent[i] = np.sqrt(closest)

    # remove parents with distance > max_dist
    # too_far = dist_parent[:] > max_dist
    # parent[too_far] = np.arange(n)[too_far]
    for i in range(n):
        if dist_parent[i] > max_dist:
            parent[i] = i
    old = np.zeros_like(parent)

    # flatten forest (mark each point with root of corresponding tree)
    while (old != parent).any():
        #old = parent
        #parent = parent[parent]
        for i in range(n):
            old[i] = parent[i]
            parent[i] = parent[parent[i]]

    parent = np.unique(parent, return_inverse=True)[1]

    if return_tree:
        return parent, dist_parent
    return parent
