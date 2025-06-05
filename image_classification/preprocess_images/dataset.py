import torch
import torch.nn.functional as F
import os
from torch.utils.data import Dataset
from preprocess import normalize_image
from torchvision.io import decode_image
import pandas as pd

class LIVE(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
            self.csv_file = pd.read_csv(csv_file)
            self.img_dir = img_dir
            self.transform = transform

    def __len__(self):
        return len(self.csv_file)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.img_lab.at[index, 'image_path'])
        image = decode_image(img_path)
        label = self.csv_file.at(index, 'dmos_reverse_normalized')
        if self.transform:
             image = self.transform(image)

        return image,label

def get_dataset(dataset:str = 'live'):
    if dataset == 'live':
        root_path = '../data/Live_IQA_release2/'
        csv_path = root_path + 'my_dmos_norm.csv'

        # Create the dataset
        dataframe = pd.read_csv(csv_path, sep=';')
        dataframe.pop('distorcion')
        dataframe.pop('index')
        dataframe.pop('ref_image_path')
        dataframe.pop('dmos')
        dataframe.pop('dmos_new')
        dataframe.pop('dmos_std')
        dataframe.pop('dmos_normalized')

        # target = dataframe.pop('dmos_reverse_normalized')
        # dataframe['image_path'] = dataframe['image_path'].apply(lambda x: root_path + x)

        live_dataset = LIVE(dataframe, root_path, normalize_image)

        return live_dataset
    
    else:
        print("Ainda não existe suporte para esse dataset :(")


if __name__ == '__main__':
    a = get_dataset("live")