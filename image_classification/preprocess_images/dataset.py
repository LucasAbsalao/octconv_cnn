import torch
import torch.nn.functional as F
import pandas as pd

def load_dataset(dataset:str = 'live'):
    if dataset == 'live':
        root_path = '/content/drive/MyDrive/Datasets/Live_IQA_release2/'
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

        target = dataframe.pop('dmos_reverse_normalized')
        dataframe['image_path'] = dataframe['image_path'].apply(lambda x: root_path + x)

        ds = tf.data.Dataset.from_tensor_slices((dataframe.values, target.values))

        return ds
    
    else:
        print("ainda não existe suporte para esse dataset :(")