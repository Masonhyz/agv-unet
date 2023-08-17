from skimage.measure import label
from scipy.spatial.distance import directed_hausdorff
import numpy as np


def hausdorff_distance(mask1, mask2):
    return max(directed_hausdorff(mask1, mask2)[0], directed_hausdorff(mask2, mask1)[0])


mask1 = np.array([
    [0, 1],
    [1, 0]
])

mask2 = np.array([
    [1, 0],
    [0, 1]
])

mask3 = np.array([
    [1, 1, 1],
    [1, 1, 0],
    [1, 1, 1]
])

mask4 = np.array([
    [0, 1, 1],
    [1, 1, 0],
    [1, 1, 1]
])


array1 = np.random.choice([0, 1], size=(6, 6))
array2 = np.random.choice([0, 1], size=(6, 6))
array3 = np.random.choice([0, 1], size=(6, 6))

print(hausdorff_distance(array3, array1))  # Output may vary



def is_one_piece_mask(binary_mask):
    labeled_mask = label(binary_mask, connectivity=1)
    unique_labels = set(labeled_mask.flatten())
    num_labels = len(unique_labels) - (1 if 0 in unique_labels else 0)
    return num_labels == 1


import numpy as np

diagonal_mask = np.array([
    [1, 1, 1],
    [1, 1, 0],
    [1, 1, 1]
])


# print(is_one_piece_mask(diagonal_mask))

