import numpy as np

def generate_synthetic_image():
    
    image_test = np.random.normal(size=(4,5,3))
    mask_test = np.array([
        [0,0,1,1,1],
        [0,0,1,1,1],
        [0,0,1,1,1],
        [0,0,1,1,1]
    ])
    
    return image_test, mask_test
