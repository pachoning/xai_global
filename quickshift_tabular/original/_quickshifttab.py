import numpy as np

#from .._shared.filters import gaussian
#from .._shared.utils import _supported_float_type
#from ..color import rgb2lab
#from ..util import img_as_float
import os
print(os.getcwd())
from _quickshifttab_cy import _quickshifttab_cython


def quickshifttab(X, ratio=1.0, kernel_size=5, max_dist=10,
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
    X = np.ascontiguousarray(X)

    cluster_list = _quickshifttab_cython(
        X, kernel_size=kernel_size, max_dist=max_dist,
        return_tree=return_tree, random_seed=random_seed)
    return cluster_list