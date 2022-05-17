import numpy as np
import pandas as pd
import cv2
import os
import random
from tqdm import tqdm
import timm
import torch
import torch.nn as nn

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

def predict(model_name, dataset, model_path):
    model = CustomConvNext(n_classes=n_classes, model_name=model_name)
    model.load_state_dict(torch.load(model_path))
    model = model.cuda(device)
    model.eval()
    tqdm_dataset = tqdm(enumerate(dataset))
    training = False
    results = []
    scores = []
    for batch, batch_item in tqdm_dataset:
        img = batch_item['img'].to(device)
        with torch.no_grad():
            output = model(img)
        output = torch.softmax(output, 1).squeeze(0)
        pred = torch.argmax(output).detach().cpu().numpy()
        scores.append(output)
        results.append(pred)
    return results, scores

if __name__ == "__main__":
    device = torch.device("cuda")
    n_classes = len(df['label'].unique())
    img_size = 512

    sub_name = 'convnext'

    test_dataset = CustomDataset(test_df['file_name'], labels=None, mode='test', img_size=img_size)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=0, shuffle=False)

    tta_dataset = CustomDataset(test_df['file_name'], labels=None, mode='tta', img_size=img_size)
    tta_dataloader = torch.utils.data.DataLoader(tta_dataset, batch_size=1, num_workers=0, shuffle=False)

    preds, scores = predict('convnext_large_384_in22ft1k', test_dataloader, 'models/convnext_large_384_in22ft1k_last.pt')

    submission = pd.read_csv('sample_submission.csv')
    preds_name =  [np.unique(df['label'])[pred] for pred in preds]
    submission.iloc[:,1] = preds_name
    submission.to_csv(f'{sub_name}_test.csv', index=False) # 단일 모델 inference 결과

    scores_csv = open(f'{sub_name}_test_scores.csv', 'w') # Ensemble용
    for i, score in enumerate(scores):
        scores_csv.write(f'{i}')
        for s in score:
            scores_csv.write(f',{s.item()}')
        scores_csv.write('\n')
    scores_csv.close()

    
