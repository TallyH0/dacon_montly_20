from efficientnet_pytorch import EfficientNet
import torch
import torch.nn as nn
import os
from tqdm import tqdm
import cv2
import albumentations as A
import numpy as np


FEATURE_DIM = [1280, 1280, 1408, 1536, 1792, 2048, 2304, 2560]
IMAGE_SIZE = [224, 240, 260, 300, 380, 456, 528, 600]
DROP_OUT = [0.2, 0.24, 0.28, 0.32, 0.37, 0.41, 0.45, 0.5]


def imread_unicode(path):
    stream = open(path, 'rb')
    byte = bytearray(stream.read())
    np_array = np.asarray(byte, dtype=np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_UNCHANGED)
    
    return img


class General_dataset(torch.utils.data.Dataset):
    def __init__(self, path_csv, height, width, aug=None, in_memory = False):
        self.aug = aug
        self.data = []
        self.labels = []
        self.label_dict = []
        self.products = []
        self.product_dict = []
        self.height = height
        self.width = width
        self.in_memory = in_memory
        
        with open(path_csv) as f:
            f.__next__()
            for line in tqdm(f):
                index, file_name, class_, state, label = line.strip().split(',')
                self.labels.append(label)
                self.products.append(class_)
                if in_memory:
                    path = os.path.join('train', file_name)
                    img = cv2.imread(path)
                    img = cv2.resize(img, (width, height))
                    self.data.append(img)
                else:
                    self.data.append(os.path.join('train', file_name))

        ids = np.unique(self.labels)
        self.label_dict = list(ids)
        self.product_dict = list(np.unique(self.products))
        
    def __getitem__(self, idx):

        if self.in_memory:
            img = self.data[idx]
        else:
            path = self.data[idx]
            img = cv2.imread(path)
            img = cv2.resize(img, (self.width, self.height))

        label = self.label_dict.index(self.labels[idx])
        label_product = self.product_dict.index(self.products[idx])
        if self.aug:
            aug = self.aug(image=img)
            img = aug['image']
        
        return img, label, label_product
    
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

def hierachy_prob(logit, hierachy):
    prob_hierachy = []
    prob = torch.softmax(logit, dim=1)
    for index in hierachy:
        prob_hierachy.append(
            prob[:, index].sum(dim=1)
        )
    
    return torch.stack(prob_hierachy).permute(1, 0)


class EfficientNet_(nn.Module):
    def __init__(self, name, dropout, num_class):
        super().__init__()
        self._backbone = Load_pretrained_backbone(name)
        self._dropout = nn.Dropout(dropout)
        self._fc = nn.Linear(FEATURE_DIM[int(name[-1])], num_class)
    
    def forward(self, x):
        x = self._backbone(x)
        x = x.flatten(start_dim=1)
        x = self._dropout(x)
        x = self._fc(x)
        
        return x
        

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1, reduction='mean'):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim
        self.reduction = reduction

    def forward(self, pred, target, is_prob = False):
        if is_prob:
            pred = torch.log(pred)
        else:
            pred = pred.log_softmax(dim=self.dim)

        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        
        if self.reduction == 'mean':
            return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
        else:
            return torch.sum(-true_dist * pred, dim=self.dim)


def train(name, path_model, loader_train, loss_fn, max_epoch=100, lr=1e-3, hierachy = []):
    net = EfficientNet_(name, DROP_OUT[int(name[-1])], 88)

    if hierachy:
        loss_fn_hierachy = LabelSmoothingLoss(15, 0.1)

    net.cuda()
    net = nn.DataParallel(net)
    optim = torch.optim.AdamW(net.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optim, lr, epochs=max_epoch, steps_per_epoch=len(loader_train))
    for i in range(max_epoch):
        print('Epoch : %d' %(i+1))
        loss_train = 0
        net.train()
        for x, y, y_product in tqdm(loader_train):
            optim.zero_grad()

            x = x.to(torch.float32).cuda().permute(0, 3, 1, 2) / 255.0
            y = y.to(torch.int64).cuda()
            y_product = y_product.to(torch.int64).cuda()
            y_ = net(x)

            loss = loss_fn(y_, y)
            if hierachy:
                prob_product = hierachy_prob(y_, hierachy)
                loss_hierachy = loss_fn_hierachy(prob_product, y_product, True)
                loss += loss_hierachy

            loss_train += loss.item()

            loss.backward()
            optim.step()
            scheduler.step()


        print('loss_train :', loss_train / len(loader_train))
    
    torch.save(net.state_dict(), path_model)

if __name__ == '__main__':

    ##Parameters
    model_name = 'efficientnet-b5'
    batch_size = 64
    lr = 1e-3
    max_epoch = 100
    num_workers = 8
    path_output = 'model_b5_new_aug_smoothing_hierachy'
    label_smoothing = True
    flag_hierachy = True

    imgh, imgw = IMAGE_SIZE[int(model_name[-1])], IMAGE_SIZE[int(model_name[-1])]
    hole_h = int(0.1 * imgh)
    hole_w = int(0.1 * imgw)
    aug = A.Compose([
        A.Affine(rotate=(-180, 180)),
        A.ColorJitter(hue=0.0),
        A.CLAHE(),
        A.CoarseDropout(max_holes=2, max_height = hole_h, max_width = hole_w),
    ])

    dataset_train = General_dataset('train_df.csv', imgh, imgw, aug, True)
    hierachy = []
    if flag_hierachy:
        hierachy = [[] for i in range(len(dataset_train.product_dict))]
        for i, name in enumerate(dataset_train.label_dict):
            index = dataset_train.product_dict.index(name.split('-')[0])
            hierachy[index].append(i)


    sample_weight = []
    _, label_count = np.unique(dataset_train.labels, return_counts=True)
    label_count = 1 / label_count
    for label in dataset_train.labels:
        sample_weight.append(label_count[dataset_train.label_dict.index(label)])
        
    sampler = torch.utils.data.WeightedRandomSampler(sample_weight, len(sample_weight))
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size, sampler=sampler, num_workers=num_workers)

    if label_smoothing:
        loss_fn = LabelSmoothingLoss(88, 0.1)
    else:
        loss_fn = nn.CrossEntropyLoss()


    for i in range(5):
        path_model = path_output + '_%02d.pth' % i
        train(model_name, path_model, loader_train, loss_fn, max_epoch, lr, hierachy)

