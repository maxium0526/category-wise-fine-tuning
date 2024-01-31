import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
import os
from tqdm import tqdm
from models import Model
import config

def generate_FT_dataset(backbone, dataloader, device='cuda'):
    zs , ys = [], []

    backbone.eval()
    with torch.no_grad():
        for batch, (x, y) in tqdm(enumerate(dataloader), total=dataloader.__len__()):
            x, y = x.to(device), y.to(device)
            z = backbone(x)
            zs.append(z)
            ys.append(y)
    zs = torch.cat(zs)
    ys = torch.cat(ys)

    return zs, ys

if __name__=='__main__':
    output_dir = 'output/CFT'

    batch_size = 32

    train_transform = transforms.Compose([
        # transforms.AutoAugment(interpolation=torchvision.transforms.functional.InterpolationMode.BICUBIC),
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
    ])

    valid_transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
    ])

    train_dataset = config.train_dataset
    train_dataset.transform = train_transform

    train_dataset.df.to_csv('prepare_train.csv')

    valid_dataset = config.valid_dataset
    valid_dataset.transform = valid_transform

    valid_dataset.df.to_csv('prepare_valid.csv')

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size)

    num_categories = train_dataset.num_categories

    state_dict = torch.load(os.path.join('output', 'train', 'best.pth'))
    model = Model(num_categories)
    model.load_state_dict(state_dict)
    model = model.to('cuda')
    model.eval()

    backbone = model.backbone

    W = model.classification_layer.weight
    b = model.classification_layer.bias

    W, b = W.cpu(), b.cpu()

    z_train, y_train = generate_FT_dataset(backbone, train_dataloader)
    z_valid, y_valid = generate_FT_dataset(backbone, valid_dataloader)

    z_train, y_train = z_train.cpu(), y_train.cpu()
    z_valid, y_valid = z_valid.cpu(), y_valid.cpu()

    Path(os.path.join('output', 'CFT', 'original')).mkdir(parents=True, exist_ok=True)
    torch.save(W, os.path.join('output', 'CFT', 'original', 'weight.pt'))
    torch.save(b, os.path.join('output', 'CFT', 'original', 'bias.pt'))

    Path(os.path.join('output', 'CFT', 'cache')).mkdir(parents=True, exist_ok=True)
    torch.save(z_train, os.path.join('output', 'CFT', 'cache', 'z_train.pt'))
    torch.save(y_train, os.path.join('output', 'CFT', 'cache', 'y_train.pt'))
    torch.save(z_valid, os.path.join('output', 'CFT', 'cache', 'z_valid.pt'))
    torch.save(y_valid, os.path.join('output', 'CFT', 'cache', 'y_valid.pt'))