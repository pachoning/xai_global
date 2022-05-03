import os
from skimage import io
from skimage.measure import EllipseModel
import numpy as np
import pandas as pd
from functools import partial

class FeaturesManager:
    
    def __init__(self, img, image_name, mask):

        self.img = img
        self.image_name = image_name
        self.mask = mask
        self.features_properties = [
            #{'name': 'image_name', 'multiple_values': False, 'fun': self.get_image_name},
            #{'name': 'num_superpixel', 'multiple_values': False, 'fun': self.get_num_superpixel},
            {'name': 'total_pixels', 'multiple_values': False, 'fun': self.get_total_pixels},
            {'name': 'mean_channel', 'multiple_values': True, 'fun': self.get_mean},
            {'name': 'std_channel', 'multiple_values': True, 'fun': self.get_std},
            {'name': 'channels_correlation', 'multiple_values': True, 'fun': self.get_channels_correlation},
            {'name': 'mass_center', 'multiple_values': True, 'fun': self.get_mass_center},
            {'name': 'ellipse_ratio', 'multiple_values': False, 'fun': self.get_ellipse_features},
            
            #{'name': 'q_5_channel', 'multiple_values': True, 'fun': partial(self.get_quant, q=.05, per_axis=True)},
            #{'name': 'q_10_channel', 'multiple_values': True, 'fun': partial(self.get_quant, q=.1, per_axis=True)},
            #{'name': 'q_25_channel', 'multiple_values': True, 'fun': partial(self.get_quant, q=.25, per_axis=True)},
            #{'name': 'q_50_channel', 'multiple_values': True, 'fun': partial(self.get_quant, q=.5, per_axis=True)},
            #{'name': 'q_75_channel', 'multiple_values': True, 'fun': partial(self.get_quant, q=.75, per_axis=True)},
            #{'name': 'q_90_channel', 'multiple_values': True, 'fun': partial(self.get_quant, q=.9, per_axis=True)},
            #{'name': 'q_95_channel', 'multiple_values': True, 'fun': partial(self.get_quant, q=.95, per_axis=True)},
            
            {'name': 'q_5', 'multiple_values': False, 'fun': partial(self.get_quant, q=.05, per_axis=False)},
            {'name': 'q_10', 'multiple_values': False, 'fun': partial(self.get_quant, q=.1, per_axis=False)},
            {'name': 'q_25', 'multiple_values': False, 'fun': partial(self.get_quant, q=.25, per_axis=False)},
            {'name': 'q_50', 'multiple_values': False, 'fun': partial(self.get_quant, q=.5, per_axis=False)},
            {'name': 'q_75', 'multiple_values': False, 'fun': partial(self.get_quant, q=.75, per_axis=False)},
            {'name': 'q_90', 'multiple_values': False, 'fun': partial(self.get_quant, q=.9, per_axis=False)},
            {'name': 'q_95', 'multiple_values': False, 'fun': partial(self.get_quant, q=.95, per_axis=False)},
        ]
        self.mask_shape = mask.shape
        
    def get_quant(self, q, superpixel, per_axis, normalise_features, *args, **kwargs):
        if normalise_features:
            if per_axis:
                return np.quantile(a=superpixel/255, q=q, axis=0)
            else:
                return np.quantile(a=superpixel/255, q=q)
        else:
            if per_axis:
                return np.quantile(a=superpixel, q=q, axis=0)
            else:
                return np.quantile(a=superpixel, q=q)

    def get_total_pixels(self, superpixel, normalise_features, *args, **kwargs):
        total_pixels = self.mask_shape[0] * self.mask_shape[1]
        if normalise_features:
            return superpixel.shape[0]/total_pixels
        else:
            return superpixel.shape[0]
    
    def get_mean(self, superpixel, normalise_features, *args, **kwargs):
        if normalise_features:
            superpixels_norm = superpixel/255
            return np.mean(superpixels_norm, axis=0)
        else:
            return np.mean(superpixel, axis=0)
    
    def get_std(self, superpixel, normalise_features, *args, **kwargs):
        if normalise_features:
            return np.std(superpixel/255, axis=0)
        else:
            return np.std(superpixel, axis=0)
        
    def get_channels_correlation(self, superpixel, normalise_features, num_channels, *args, **kwargs):
        
        superpixel_norm = superpixel
        superpixel_shape = superpixel_norm.shape
        
        if normalise_features:
            superpixel_norm = superpixel/255
        
        if superpixel_shape[0]>1 and superpixel_shape[1]>1:
            corr_matrix = np.corrcoef(superpixel_norm, rowvar=False)
            idx = np.triu_indices(n=num_channels, m=num_channels, k=1)
            corr_vector = corr_matrix[idx]
        
        else:
            n = num_channels
            len_vector = int(n*(n-1)/2)
            corr_vector = np.repeat(None, len_vector)
            
        return corr_vector
    
    def get_mass_center(self, coordinates, normalise_features, *args, **kwargs):
        
        coordinates_norm = np.asarray(coordinates, dtype=np.float64)
        if normalise_features:
            for i,len_axis in enumerate(self.mask_shape):
                coordinates_norm[i] = coordinates_norm[i]/len_axis
        return np.mean(coordinates_norm, axis=1)
        
    def get_ellipse_features(self, coordinates, *args, **kwargs):
        
        if len(coordinates[0]) > 1 and len(coordinates[1]) > 1:
            coordinates_stack = np.column_stack(coordinates)
            ellipse = EllipseModel()
            ellipse.estimate(coordinates_stack)
            xc, yc, a, b, theta = ellipse.params
            ratio = b/a
            if(b>a):
                ratio = 1/ratio
        else:
            ratio = None
        
        return ratio

    def create_data_frame(self, dict_features, list_superpixels):
        
        df = pd.DataFrame(dict_features)
            
        for feature in self.features_properties:
            if feature['multiple_values']:
                feature_name = feature['name']
                values = df[feature_name].to_list()
                len_values = len(values[0])
                names = [feature_name + str(i) for i in range(len_values)]
                df[names] = values
                df.drop(columns=feature_name, inplace=True)

        df['image_name'] = [self.image_name for x in range(len(df))]
        df['num_superpixel'] = list_superpixels
        df.set_index(['image_name', 'num_superpixel'], inplace=True)

        return df
    
    def clean_data(self, df):

        columns = df.columns
        for column in columns:
            values = df[column]
            mean_column = np.mean(values)
            idx_missing = np.isnan(values)

            if len(idx_missing)>0:
                df.loc[idx_missing, column] = mean_column        
    
    def extract_from_manual(self, normalise_features, num_channels):

        img = self.img
        mask = self.mask

        dict_features = {}
        list_superpixels = []
        
        # Obtain features
        unique_superpixels = np.unique(self.mask)
        
        for index in unique_superpixels:
            list_superpixels.append(index)
            coordinates = np.where(mask == index)
            superpixel = img[coordinates]
            num_pixels = self.get_total_pixels(superpixel=superpixel,normalise_features=normalise_features)
            
            for feature in self.features_properties:
                feature_name = feature['name']
                feature_values = feature['fun'](superpixel=superpixel, num_superpixel=index,
                                                img=img, mask=mask, coordinates=coordinates,
                                                normalise_features=normalise_features,
                                                num_channels=num_channels)

                if feature_name in dict_features.keys():
                    dict_features[feature_name].append(feature_values)
                else:
                    dict_features[feature_name] = [feature_values]

        df_features = self.create_data_frame(dict_features, list_superpixels)
        
        self.clean_data(df_features)
        
        return df_features
    
    def get_smalles_patch(self, num_superpixel):
        
        img = self.img
        mask = self.mask
        idx_mask = np.where(mask == num_superpixel)
        min_idx = np.min(idx_mask, axis=-1)
        max_idx = np.max(idx_mask, axis=-1)
        slicer = []

        for min_val, max_val in zip(min_idx, max_idx):
            slice_dim = slice(min_val, max_val+1, 1)
            slicer.append(slice_dim)

        slicer = tuple(slicer)
        sliced_image = img[slicer]
        return img[slicer]

    def extract_from_model(self, model_feature_extractor, target_shape):
        img = self.img
        mask = self.mask
        list_features = []
        list_num_superpixel = []
        list_superpixels = np.unique(mask)
        list_total_zeros = []

        for sp in list_superpixels:
            coordinates = np.where(mask == sp)
            smallest_patch = self.get_smalles_patch(num_superpixel=sp)
            smallest_patch = image.smart_resize(smallest_patch, size=target_shape)
            smallest_patch = np.expand_dims(smallest_patch, axis=0)
            features = model_feature_extractor(smallest_patch)
            np_features = features.numpy()
            np_features = np_features[0]
            total_zeros = np.sum(np_features <= 10**(-10))
            list_total_zeros.append(total_zeros)
            list_features.append(np_features)
            list_num_superpixel.append(sp)

        df_features = pd.DataFrame(np.array(list_features))
        df_features['image_name'] = [self.image_name for x in range(len(df_features))]
        df_features['num_superpixel'] = list_num_superpixel
        df_features.set_index(['image_name', 'num_superpixel'], inplace=True)
        return df_features, list_total_zeros

    def get_features(self, use_model_to_extract_features, 
                     model_feature_extractor, target_shape,
                     normalise_features, num_channels):

        if use_model_to_extract_features:
            df, list_total_zeros = self.extract_from_model(model_feature_extractor, target_shape)
        else:
            list_total_zeros = []
            df = self.extract_from_manual(normalise_features=normalise_features, num_channels=num_channels)
        return df, list_total_zeros
