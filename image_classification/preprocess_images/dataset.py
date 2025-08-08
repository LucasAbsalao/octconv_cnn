import torch
import torch.nn.functional as F
import os
from PIL import Image
from torch.utils.data import Dataset
from .preprocess import normalize_image
import torchvision.transforms as transforms
import pandas as pd

class LIVE(Dataset):    
    def __init__(self, csv_file, img_dir, transform=None):
            self.csv_file = csv_file
            self.img_dir = img_dir
            self.transform = transform

    def __len__(self):
        return len(self.csv_file)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.csv_file.at[index, 'image_path'])
        image = Image.open(img_path).convert('RGB')
        totensor = transforms.ToTensor()
        image = totensor(image)
        label = self.csv_file.at[index, 'dmos_reverse_normalized']
        label = torch.tensor(label, dtype=torch.float32)
        if self.transform is not None:
             image = self.transform(image)

        return (image,label)
    
class KonIQ(Dataset):    
    def __init__(self, csv_file, img_dir, transform=None):
            self.csv_file = csv_file
            self.img_dir = img_dir
            self.transform = transform

    def __len__(self):
        return len(self.csv_file)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.csv_file.at[index, 'image_name'])
        image = Image.open(img_path).convert('RGB')
        totensor = transforms.ToTensor()
        image = totensor(image)
        label = self.csv_file.at[index, 'MOS_zscore']
        label = torch.tensor(label, dtype=torch.float32)

        return (image,label)

def custom_collate_fn(batch):
    image = [data[0] for data in batch]
    labels = [data[1] for data in batch]
    labels = torch.tensor(labels)
    return [image,labels]



def get_dataset(dataset:str = 'live', path:str = None, img_path:str = None, preprocess:bool = True):
    if dataset == 'live':
        if path is None:
            root_path = '../data/Live_IQA_release2/'
        else:
            root_path = path
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
        if preprocess:
            dataset_final = LIVE(dataframe, root_path, normalize_image)
        else:
            dataset_final = LIVE(dataframe, root_path)
    
    elif dataset == 'koniq-10k':
        if path is None:
            root_path = '../data/KonIQ-10k/'
            img_path = '../data/KonIQ-10k/512x384/'
        else:
            root_path = path
            img_path = img_path

        csv_path = root_path + 'koniq10k_scores_and_distributions.csv'

        dataframe = pd.read_csv(csv_path)
        dataframe = dataframe[['image_name', 'MOS_zscore']]
        dataframe['MOS_zscore'] = dataframe['MOS_zscore']/100.0

        if preprocess:
            dataset_final = KonIQ(dataframe, img_path, normalize_image)
        else:
            dataset_final = KonIQ(dataframe, img_path)
    else:
        print("Ainda não existe suporte para esse dataset :(")

    return dataset_final

if __name__ == '__main__':
    a = get_dataset("live")
    dataloader = torch.utils.data.DataLoader(a, batch_size=2, shuffle= True, collate_fn=custom_collate_fn, num_workers=2)
    print("loaded")
    for image, label in dataloader:
        if torch.max(label)>1.0:
            print(torch.max(label))