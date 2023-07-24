from PIL import Image
import numpy as np
import torch.utils.data as da
import os
import torch


class KRD(da.Dataset):
    """
    Custom class for the knee recess distension dataset.
    """
    def __init__(self, config):
        """
        Precondition: there has to be an Image directory in root_dir, and a Mask
        directory as well, both filled with png files with corresponding names.
        :param config: dict, an attribute of Configs instance
        """
        self.root_dir = config["root dir"]
        self.us_dir = os.path.join(self.root_dir, 'US')
        self.masks_dir = os.path.join(self.root_dir, 'Masks')
        self.filenames = os.listdir(self.us_dir)
        self.config = config

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        """
        get the idx-th item in the dataset
        :param idx: the index of the item of interest
        :return: a tuple
        at index 0: the file name of the image
        at index 1: the tensor of the ultrasound image of shape (3, 256, 256)
        at index 2: the tensor of the segmentation mask of shape (1, 256, 256)
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        s = self.config["image size"]
        # Get filenames
        file_name = self.filenames[idx]
        us_file_path = os.path.join(self.us_dir, file_name)
        mask_file_path = os.path.join(self.masks_dir, file_name)

        # Load images
        us_image = Image.open(us_file_path).convert('RGB').\
            resize((s, s))
        mask_image = Image.open(mask_file_path).convert('L').resize((s, s))
        us_np = np.array(us_image)
        mask_np = np.array(mask_image)
        us_tensor = torch.from_numpy(us_np).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).float() / 255.0

        return file_name, us_tensor, mask_tensor
