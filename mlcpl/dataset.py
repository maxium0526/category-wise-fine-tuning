import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import os
import pandas as pd
from PIL import Image
import xmltodict
import json
import glob
from tqdm import tqdm
import time

class MLCPLDataset(Dataset):
    def __init__(self, dataset_path, df, num_categories, transform):
        self.dataset_path = dataset_path
        self.df = df
        self.num_categories = num_categories
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.dataset_path, row['Path'])
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        pos_category_nos = json.loads(row['Positive'].replace(';', ','))
        neg_category_nos = json.loads(row['Negative'].replace(';', ','))
        unc_category_nos = json.loads(row['Uncertain'].replace(';', ','))
        target = to_one_hot(self.num_categories, np.array(pos_category_nos), np.array(neg_category_nos), np.array(unc_category_nos))
        return img, target

    def test(self):
        return self.__getitem__(0)

    def get_samples(self, indices):
        samples = []
        if indices is None:
            sub_df = self.df
        else:
            sub_df = self.df.iloc[indices]
        for i, row in sub_df.iterrows():
            print(f'Loading {i+1}/{sub_df.shape[0]}', end='\r')
            samples.append(self.__getitem__(i))
        return samples

def fill_nan_to_negative(old_records, num_categories):
    new_records = []
    for (i, path, pos_category_nos, neg_category_nos, unc_category_nos) in old_records:
        new_neg_category_nos = [x for x in range(num_categories) if x not in pos_category_nos+unc_category_nos]
        new_records.append((i, path, pos_category_nos, new_neg_category_nos, unc_category_nos))
    return new_records

def drop_labels(old_records, target_partial_ratio, seed=526):
    rng = np.random.Generator(np.random.PCG64(seed=seed))

    new_records = []
    for (i, path, pos_category_nos, neg_category_nos, unc_category_nos) in old_records:
        new_pos_category_nos = [no for no in pos_category_nos if rng.random() < target_partial_ratio]
        new_neg_category_nos = [no for no in neg_category_nos if rng.random() < target_partial_ratio]
        new_unc_category_nos = [no for no in unc_category_nos if rng.random() < target_partial_ratio]
        new_records.append((i, path, new_pos_category_nos, new_neg_category_nos, new_unc_category_nos))
    return new_records

def to_one_hot(num_categories, pos_category_nos, neg_category_nos, unc_category_nos):
    one_hot = torch.full((num_categories, ), torch.nan, dtype=torch.float32)
    one_hot[pos_category_nos] = 1.0
    one_hot[neg_category_nos] = 0.0
    one_hot[unc_category_nos] = -1.0
    return one_hot

def records_to_df(records):
    records = [(i, path, json.dumps(pos_category_nos).replace(',', ';'), json.dumps(neg_category_nos).replace(',', ';'), json.dumps(unc_category_nos).replace(',', ';')) for (i, path, pos_category_nos, neg_category_nos, unc_category_nos) in records]
    df = pd.DataFrame(records, columns=['Id', 'Path', 'Positive', 'Negative', 'Uncertain'])
    return df

def MSCOCO(dataset_path, year='2014', split='train', partial_ratio=1.0, transform=transforms.ToTensor()):
    from pycocotools.coco import COCO

    if split == 'train':
        subset = 'train'
    if split == 'valid':
        subset = 'val'

    coco = COCO(os.path.join(dataset_path, 'annotations', f'instances_{subset}{year}.json'))
    all_category_ids = coco.getCatIds()
    num_categories = len(all_category_ids)

    records = []
    image_ids = coco.getImgIds()
    for i, img_id in enumerate(image_ids):
        print(f'Loading row: {i+1} / {len(image_ids)}', end='\r')
        img_filename = coco.loadImgs(img_id)[0]['file_name']
        path = os.path.join(subset+year, img_filename)
        pos_category_ids = [coco.loadAnns(annotation_id)[0]['category_id'] for annotation_id in coco.getAnnIds(imgIds=img_id)]
        pos_category_ids = list(set(pos_category_ids))
        pos_category_nos = [all_category_ids.index(category_id) for category_id in pos_category_ids]
        pos_category_nos.sort()
        records.append((img_id, path, pos_category_nos, [], []))
    
    records = fill_nan_to_negative(records, num_categories)
    records = drop_labels(records, partial_ratio)

    return MLCPLDataset(dataset_path, records_to_df(records), num_categories, transform)
