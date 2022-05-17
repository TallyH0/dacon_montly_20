import numpy as np
import pandas as pd
import cv2
import os
from tqdm import tqdm
import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2


from labels import label_dict

# general global variables
DATA_PATH = './'
TRAIN_PATH = './train'
TEST_PATH = './test'

df = pd.read_csv(os.path.join(DATA_PATH, 'train_df.csv'))
train_df = df.copy()
valid_df = df.copy()
test_df = pd.read_csv(os.path.join(DATA_PATH, 'test_df.csv'))

class CustomConvNext(nn.Module):
    def __init__(self, n_classes, model_name='convnext_xlarge_384_in22ft1k', pretrained=True, drop_rate=0., drop_path_rate=0.):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, drop_rate=drop_rate, drop_path_rate=drop_path_rate)
        n_features = self.model.num_features
        self.model.head.fc = nn.Linear(n_features, n_classes)

    def forward(self, x):
        x = self.model(x)
        return x

def ATransform(sample, WIDTH, HEIGHT, p=1.0):
    hole_h = int(0.1 * HEIGHT)
    hole_w = int(0.1 * WIDTH)
    
    augs = A.Compose([
        A.Affine(rotate=(-180, 180)),
        A.ColorJitter(hue=0.0),
        A.CLAHE(),
        A.CoarseDropout(max_holes=2, max_height=hole_h, max_width=hole_w)
    ], p=p)
    
    sample['img'] = augs(image=sample['img'].astype('uint8'))['image']
    
    return sample

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, files, labels=None, mode='train', img_size=528):
        self.img_size = img_size
        self.mode = mode
        self.files = files
        if mode == 'train' or mode == 'valid':
            self.labels = labels
            
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, i):
        
        if self.mode == 'train':
            label = label_dict[self.labels[i]]
            img = self.load_image(os.path.join(TRAIN_PATH, self.files[i]))
            HEIGHT, WIDTH, _ = img.shape
            sample = {'img': img, 'annot': label}
            sample = ATransform(sample, WIDTH, HEIGHT, p=1.0)
            sample = self.base(sample)
            return sample
        elif self.mode == 'valid':
            label = label_dict[self.labels[i]]
            img = self.load_image(os.path.join(TRAIN_PATH, self.files[i]))
            sample = {'img': img, 'annot': label}
            sample = self.base(sample)
            return sample
        else:
            img = self.load_image(os.path.join(TEST_PATH, self.files[i]))
            sample = {'img': img, 'annot': -1}
            sample = self.base(sample)
            return sample
        
    def load_image(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img.astype(np.float32)
    
    def base(self, sample):
    
        image, annots = sample['img'], sample['annot']

        image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

        augs = []
        augs.append(A.Normalize(always_apply=True))
        augs.append(ToTensorV2())

        sample['img'] = A.Compose(augs)(image=image)['image']
        sample['annot'] = torch.tensor(sample['annot'], dtype=torch.long)

        return sample

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))

def train_step(model, batch_item, training):
    img = batch_item['img'].to(device)
    label = batch_item['annot'].to(device)
    if training is True:
        model.train()
        optimizer.zero_grad()
        output = model(img)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        
        return loss
    else:
        model.eval()
        with torch.no_grad():
            output = model(img)
            loss = criterion(output, label)
            
        return loss

def train(model_name, model, train_dataloader, val_dataloader):
    loss_plot, val_loss_plot = [], []
    os.makedirs('models', exist_ok=True)

    for epoch in range(epochs):
        total_loss, total_val_loss = 0, 0
        
        tqdm_dataset = tqdm(enumerate(train_dataloader))
        training = True
        for batch, batch_item in tqdm_dataset:
            batch_loss = train_step(model, batch_item, training)
            total_loss += batch_loss

            tqdm_dataset.set_postfix({
                'Epoch': epoch + 1,
                'Loss': '{:06f}'.format(batch_loss.item()),
                'Total Loss' : '{:06f}'.format(total_loss/(batch+1))
            })
        scheduler.step()
            
        loss_plot.append(total_loss/(batch+1))
        
        tqdm_dataset = tqdm(enumerate(val_dataloader))
        training = False
        for batch, batch_item in tqdm_dataset:
            batch_loss = train_step(model, batch_item, training)
            total_val_loss += batch_loss

            tqdm_dataset.set_postfix({
                'Epoch': epoch + 1,
                'Val Loss': '{:06f}'.format(batch_loss.item()),
                'Total Val Loss' : '{:06f}'.format(total_val_loss/(batch+1))
            })
        total_val_loss = total_val_loss.detach().cpu().numpy()
        val_loss_plot.append(total_val_loss/(batch+1))

        if np.min(val_loss_plot) == val_loss_plot[-1]:
            torch.save(model.state_dict(), os.path.join('models', f'{model_name}_best.pt'))
        if epoch % 50 == 0:
            torch.save(model.state_dict(), os.path.join('models', f'{model_name}_{epoch}.pt'))
            
    torch.save(model.state_dict(), os.path.join('models', f'{model_name}_last.pt'))
            
    return model

if __name__ == "__main__":
    device = torch.device("cuda")
    batch_size = 16
    n_classes = len(df['label'].unique())
    learning_rate = 1e-3
    epochs = 250
    img_size = 384

    class_counts = np.unique(df['label'], return_counts=True)[1]
    num_samples = sum(class_counts)
    class_weights = [num_samples / class_counts[i] for i in range(len(class_counts))]
    weights = [class_weights[label_dict[df['label'].values[i]]] for i in range(int(num_samples))]
    sampler = torch.utils.data.WeightedRandomSampler(torch.DoubleTensor(weights), int(num_samples))

    train_dataset = CustomDataset(train_df['file_name'].values, train_df['label'].values, mode='train', img_size=img_size)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=8, sampler=sampler)

    val_dataset = CustomDataset(valid_df['file_name'].values, valid_df['label'].values, mode='valid', img_size=img_size)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=8, shuffle=False)

    model_name = 'convnext_large_384_in22ft1k'
    model = CustomConvNext(n_classes, model_name=model_name, pretrained=True, drop_rate=0.45, drop_path_rate=0.2)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = LabelSmoothingLoss(classes=n_classes, smoothing=0.1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, learning_rate, epochs=epochs, steps_per_epoch=len(train_dataloader))

    trained_model = train(model_name, model, train_dataloader, val_dataloader)

