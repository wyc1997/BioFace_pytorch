import torch
from torch.utils.data import DataLoader, Dataset
import h5py
import pandas as pd
import os


class MaskedCelebADataset(Dataset):
    def __init__(self, folder_name, meta_file, training=True):
        super(MaskedCelebADataset, self).__init__()
        self.folder_name = folder_name
        if training:
            meta_file = "train_" + meta_file
        else:
            meta_file = "test_" + meta_file
        self.meta = pd.read_csv(os.path.join(folder_name, meta_file))

    def __len__(self):
        return self.meta.shape[0]

    def __getitem__(self, idx):
        file_meta = self.meta.iloc[idx,:]
        file_directory = os.path.join(self.folder_name, file_meta["file_name"])
        file = h5py.File(file_directory, "r")

        data = file['zx_7'][file_meta["file_index"]]
        image = torch.tensor(data[:3, :, :])
        mask = torch.tensor(data[6, :, :])
        masked_image = mask * image

        normal = torch.tensor(data[3:6, :, :])
        shp = torch.tensor(data[7:10, :, :])
        shading = torch.sum(torch.mul(normal, shp), 0)
        shading = mask * shading

        return {"image": image, "shading": shading, "mask":mask}
