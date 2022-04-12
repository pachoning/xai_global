#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

import numpy as np
cimport numpy as cnp

from fused_numerics cimport np_floats

from libc.math cimport exp, sqrt, ceil
from libc.float cimport DBL_MAX

cnp.import_array()


def _quickshifttab_cython(np_floats[:, :] X, np_floats kernel_size,
                       np_floats max_dist, bint return_tree, int random_seed):
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

    if np_floats is cnp.float64_t:
        dtype = np.float64
    else:
        dtype = np.float32

    # TODO join orphaned roots?
    # Some nodes might not have a point of higher density within the
    # search window. We could do a global search over these in the end.
    # Reference implementation doesn't do that, though, and it only has
    # an effect for very high max_dist.

    # window size for neighboring pixels to consider
    cdef np_floats inv_kernel_size_sqr = -0.5 / (kernel_size * kernel_size)
    cdef int kernel_width = <int>ceil(3 * kernel_size)

    # cdef Py_ssize_t height = image.shape[0]
    # cdef Py_ssize_t width = image.shape[1]
    # cdef Py_ssize_t channels = image.shape[2]
    
    cdef Py_ssize_t n = X.shape[0]
    cdef Py_ssize_t p = X.shape[1] - 2

    cdef np_floats[:] densities = np.zeros(n, dtype=dtype)
    cdef np_floats[:] current_point = np.zeros(p+2, dtype=dtype)

    cdef np_floats current_density, closest, dist, t
    cdef Py_ssize_t i, j, r_min, r_max, c_min, c_max

    # this will break ties that otherwise would give us headache
    densities += random_state.normal(
        scale=0.00001, size=n
    ).astype(dtype, copy=False)

    # default parent to self
    cdef cnp.int8_t[:] parent = np.arange(n, dtype=np.intp)
    cdef np_floats[:] dist_parent = np.zeros(n, dtype=dtype)

    # compute densities
    #with nogil:
    for i in range(n):
        current_point = X[i,:]
        for i_ in range(n):
            dist = 0
            for j in range(p+2):
                t = (current_point[j]-X[i_, j])
                dist += t * t
                densities[i] += exp(dist * inv_kernel_size_sqr)

    # find nearest node with higher density
    current_point = X[i,:]
    for i in range(n):
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
        dist_parent[i] = sqrt(closest)

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