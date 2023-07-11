import os
import pickle
import torch
from torch.utils.data import Dataset
import numpy as np

def pkload(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def transform(data, label):
    data = torch.from_numpy(data)
    label = torch.tensor(label)
    return data, label



'''adni2'''
class adni2(Dataset):  # train:MCI/NC  210/118=328   valid:53/29=82
    def __init__(self, data, split, mode='train'):
        data_path = os.path.join('Data',data+'_5split_' + mode + '_' + str(split) + '_data.npy')

        label_path = os.path.join('Data',data+'_5split_' + mode + '_' + str(split) + '_label.pkl')
        self.mode = mode
        self.names, self.labels = pkload(label_path)
        self.datas = np.load(data_path)


    def __getitem__(self, item):  # original:1x130x90x6  -->to split easily: cut to 128  ->1x128x90x6
        label = self.labels[item]
        data = self.datas[item, :, :, :, :]
        data, label = transform(data, label)
        return data, label

    def __len__(self):
        return len(self.names)

    def get_num_class(self):
        num = len(np.unique(self.labels))
        return num

if __name__ == '__main__':
    train_data = adni2('sample',split=5, mode='train')
    sample,lable=train_data.__getitem__(20)
    print(sample.shape)
