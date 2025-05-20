import gc
import nibabel
from torch.utils.data import Dataset, DataLoader
import os
import torch
from config import get_args
import numpy as np
import nibabel
import pandas as pd


class DataSet(Dataset):
    def __init__(self, root_path, dir):
        
        self.root_path = root_path
        self.dir = dir
        self.image_path = os.path.join(self.root_path, dir)
        self.images = os.listdir(self.image_path)  
        #self.scores_df = pd.read_excel(excel_file)  
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

        #image_id = image_index.split('.')[0]

        if normalization == 'minmax':
            del img_max
        else:
            del img_fla, index, img_median
        gc.collect()

        return img,label

    def __len__(self):
        return len(self.images)


def load_data_remove(args, root_path, path1, path2):
    train_AD = DataSet(root_path, path1)
    train_CN = DataSet(root_path, path2)
    #train_PMCI = DataSet(root_path, path3, excel_file)
    #train_SMCI = DataSet(root_path, path4, excel_file)
    #trainDataset = train_PMCI+ train_SMCI
    trainDataset = train_AD + train_CN
    train_loader = DataLoader(trainDataset, batch_size=args.batch_size, shuffle=True)
    del trainDataset
    gc.collect()
    return train_loader

