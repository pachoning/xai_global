import os
from skimage import segmentation, io

class MaskManager:
    
    def __init__(self, location, image_name):
        self.location = location
        self.image_name = image_name
        
    def get_mask(self, kernel_size, max_dist, ratio):
        
        full_path = os.path.join(self.location, self.image_name)
        img = io.imread(full_path)
        mask = segmentation.quickshift(img, kernel_size=kernel_size, max_dist=max_dist, ratio=ratio)
            
        return mask
