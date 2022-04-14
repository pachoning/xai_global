import os
from skimage import segmentation, io

class MaskManager:

    def get_mask(self, img, kernel_size, max_dist, ratio):

        mask = segmentation.quickshift(img, kernel_size=kernel_size, max_dist=max_dist, ratio=ratio)
        return mask
