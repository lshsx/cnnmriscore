import gc
import nibabel
from torch.utils.data import Dataset, DataLoader
import os
import torch
from config import get_args
import numpy as np
import nibabel
import pandas as pd
from collections import Counter
import random


class DataSet(Dataset):
    def __init__(self, root_path, dir, excel_file, oversampling=False, rate=0.5):

        self.root_path = root_path
        self.dir = dir
        self.image_path = os.path.join(self.root_path, dir)
        self.images = os.listdir(self.image_path)  
        print(excel_file)
        self.scores_df = pd.read_excel(excel_file)  

        if oversampling:

            self.apply_oversampling(rate)
    
    def apply_oversampling(self, rate):

        n_images = len(self.images)
        n_oversample = int(n_images * rate)
        oversampled = random.choices(self.images, k=n_oversample)
        self.images.extend(oversampled)

        print(f"Oversample {dir} by {n_oversample} images, total {len(self.images)} images")

        
    def __getitem__(self, index):
        label = 0  
        image_index = self.images[index]  
        img_path = os.path.join(self.image_path, image_index)  
        img = nibabel.load(img_path).get_fdata().astype('float32')  
        normalization = 'minmax'  

        if normalization == 'minmax':
            img_max = img.max()
            img = img / img_max
        elif normalization == 'median':
            img_fla = np.array(img).flatten()
            index = np.argwhere(img_fla == 0)
            img_median = np.median(np.delete(img_fla, index))
            img = img / img_median
        img = np.expand_dims(img, axis=0)

        if self.dir == 'AD/':
            label = 1
        elif self.dir == 'CN/':
            label = 0
        elif self.dir == 'PMCI/':
            label = 2
        elif self.dir == 'SMCI/':
            label = 2

        image_id = image_index[4:12].split('.')[0]
        scores_row = self.scores_df[self.scores_df['ImageID'] == image_id].iloc[0]

        adas11 = scores_row['ADAS11']
        cdrsb = scores_row['CDRSB']
        mmse = scores_row['MMSE']

        if normalization == 'minmax':
            del img_max
        else:
            del img_fla, index, img_median
        gc.collect()

        return img, adas11, cdrsb, mmse,label

    def __len__(self):
        return len(self.images)


def load_data(args, root_path, path1, path2, excel_file, oversampling=False):
    train_AD = DataSet(root_path, path1,excel_file, oversampling=oversampling)
    print(f"{path1} {len(train_AD.images)}")
    train_CN = DataSet(root_path, path2,excel_file)
    print(f"{path2} {len(train_CN.images)}")

    trainDataset = train_AD + train_CN
    train_loader = DataLoader(trainDataset, batch_size=args.batch_size, shuffle=True)
    del trainDataset
    gc.collect()
    return train_loader


def test_loading_data():
    args = get_args()
    root_path_train = "MRIbr24/train/"
    path1 = 'AD/'
    path2 = 'CN/'
    path3 = 'PMCI/'
    path4 = 'SMCI/'
    root_path_val='MRIbr24/val/'
    root_path_test = 'MRIbr24/test/'
    excel_file = 'ADlist/Allm00_score.xlsx'

    train_dataloader = load_data(args, root_path_train, path1, path2, excel_file)
    oversampled_train_dataloader = load_data(args, root_path_train, path1, path2, excel_file, oversampling=True)


if __name__ == '__main__':
    test_loading_data()

