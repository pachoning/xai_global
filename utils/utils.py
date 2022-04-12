import numpy as np
from scipy.cluster.hierarchy import dendrogram

def generate_synthetic_image():
    
    image_test = np.random.normal(size=(4,5,3))
    mask_test = np.array([
        [0,0,1,1,1],
        [0,0,1,1,1],
        [0,0,1,1,1],
        [0,0,1,1,1]
    ])
    
    return image_test, mask_test


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)