#!/usr/bin/env python
# coding: utf-8



from efficientnet_pytorch import EfficientNet
import torch
import torch.nn as nn
import os
from tqdm import tqdm
import cv2
import numpy as np

FEATURE_DIM = [1280, 1280, 1408, 1536, 1792, 2048, 2304, 2560]
IMAGE_SIZE = [224, 240, 260, 300, 380, 456, 528, 600]


def imread_unicode(path):
    stream = open(path, 'rb')
    byte = bytearray(stream.read())
    np_array = np.asarray(byte, dtype=np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_UNCHANGED)
    
    return img

def load_weight(path_weight):
    weight_dict = torch.load(path_weight)
    new_weight_dict = {}

    for key in weight_dict:
        if key[:len('module.')] != 'module.':
            return weight_dict
        else:
            new_key = key[len('module.'):]
            new_weight_dict[new_key] = weight_dict[key]
    
    return new_weight_dict



class General_dataset(torch.utils.data.Dataset):
    def __init__(self, path_csv, height, width, aug=None):
        self.aug = aug
        self.data = []
        self.labels = []
        self.label_dict = []
        self.height = height
        self.width = width
        
        with open(path_csv) as f:
            f.__next__()
            for line in f:
                index, file_name, class_, state, label = line.strip().split(',')
                self.data.append(os.path.join('train', file_name))
                self.labels.append(label)
        
        ids = np.unique(self.labels)
        self.label_dict = list(ids)
        
    def __getitem__(self, idx):
        path = self.data[idx]
        label = self.label_dict.index(self.labels[idx])
        
        img = cv2.imread(path)
        img = cv2.resize(img, (self.width, self.height))
        if self.aug:
            aug = self.aug(image=img)
            img = aug['image']
        
        return img, label
    
    def __len__(self):
        return len(self.data)


class Test_dataset(torch.utils.data.Dataset):
    def __init__(self, path_csv, height, width, aug=None):
        self.aug = aug
        self.data = []
        self.height = height
        self.width = width
        
        with open(path_csv) as f:
            f.__next__()
            for line in f:
                index, file_name= line.strip().split(',')
                self.data.append(os.path.join('test', file_name))
        
        
    def __getitem__(self, idx):
        path = self.data[idx]
        
        img = cv2.imread(path)
        img = cv2.resize(img, (self.width, self.height))
        if self.aug:
            aug = self.aug(image=img)
            img = aug['image']
        
        return img
    
    def __len__(self):
        return len(self.data)


def Load_pretrained_backbone(name='efficientnet-b7'):
    net = EfficientNet.from_pretrained(name, include_top=True)
    net_weight = net.state_dict()
    del net_weight['_fc.weight']
    del net_weight['_fc.bias']
    net = EfficientNet.from_name(name, include_top=False)
    net.load_state_dict(net_weight)
    return net


class EfficientNet_(nn.Module):
    def __init__(self, name, dropout, num_class):
        super().__init__()
        self._backbone = Load_pretrained_backbone(name)
        self._dropout = nn.Dropout(dropout)
        feature_dim = FEATURE_DIM[int(name[-1])]
        self._fc = nn.Linear(feature_dim, num_class)
    
    def forward(self, x):
        x = self._backbone(x)
        x = x.flatten(start_dim=1)
        x = self._dropout(x)
        x = self._fc(x)
        
        return x


def test(path_model, dataset_train, loader_test):
    net.load_state_dict(load_weight(path_model))
    net.to(device)
    net.eval()

    f_submssion = open(path_model + '.csv', 'w')
    f_result = open(path_model + '_result.csv', 'w')
    f_submssion.write('index,label\n')

    index = 0
    with torch.no_grad():
        for x in tqdm(loader_test):
            x = x.to(torch.float32).to(device).permute(0, 3, 1, 2) / 255.0
            logit = net(x)
            probs = torch.softmax(logit, dim=1)
            preds = torch.argmax(logit, 1)

            for pred, prob in zip(preds, probs):
                f_result.write('%d' % index)
                for p in prob:
                    f_result.write(',%f' %(p.item()))
                pred = pred.item()
                prob = prob[pred].item()
                label = dataset_train.label_dict[pred]
                f_submssion.write('%d,%s\n' %(index, label))
                f_result.write('\n')
                index += 1
    
    f_submssion.close()
        


if __name__ == '__main__':
    network_type = 'efficientnet-b5'
    net = EfficientNet_(network_type, 0, 88)
    imgh, imgw = IMAGE_SIZE[int(network_type[-1])], IMAGE_SIZE[int(network_type[-1])]

    batch_size = 1
    device = torch.device('cuda')

    dataset_train = General_dataset('train_df.csv', imgh, imgw, None)
    num_class = len(np.unique(dataset_train.labels))

    dataset_test = Test_dataset('test_df.csv', imgh, imgw)
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size)

    model_list = [
        'model_b5_new_aug_smoothing_hierachy_00.pth',
        'model_b5_new_aug_smoothing_hierachy_01.pth',
        'model_b5_new_aug_smoothing_hierachy_02.pth',
        'model_b5_new_aug_smoothing_hierachy_03.pth',
        'model_b5_new_aug_smoothing_hierachy_04.pth',
    ]

    for model_name in model_list:
        test(model_name, dataset_train, loader_test)
