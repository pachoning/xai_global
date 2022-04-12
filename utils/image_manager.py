import os
from skimage import io

class ImagesManager:

    def __init__(self, location, images_extension):
        self.location = location
        self.images_extension = images_extension

    def get_images_names(self):
        all_data = os.listdir(self.location)
        images_names = [x for x in all_data if x.split('.')[-1] == self.images_extension]
        return images_names

    def __len__(self):
        return len(self.get_images_names())

    def get_image(self, image_name):
        path_to_image = os.path.join(self.location, image_name)
        return io.imread(path_to_image)
