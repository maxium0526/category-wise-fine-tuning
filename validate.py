import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torchmetrics
import os
from tqdm import tqdm
from pathlib import Path
from mlcpl.helper import ExcelLogger
from mlcpl.metric import *
from models import Model
import config

def main():
    device = 'cuda'
    output_dir = 'output/train'
    log_dir = 'output/valid/log'

    valid_transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
    ])

    valid_dataset = config.valid_dataset
    valid_dataset.transform = valid_transform

    num_categories = valid_dataset.num_categories

    model = Model(num_categories)
    batch_size = 32

    validation_metrics = {
        'mAP': PartialMultilabelMetric(torchmetrics.functional.classification.binary_average_precision),
        'per_category_AP': PartialMultilabelMetric(torchmetrics.functional.classification.binary_average_precision, reduction=None)
    }

    ### below normally fixed
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=20, shuffle=False)
    
    model.load_state_dict(torch.load(os.path.join(output_dir, 'best.pth')))
    model = model.to(device)
    model.eval()

    excellog = ExcelLogger(os.path.join(log_dir, 'excel_log.xlsx'))

    output_dirs = os.listdir(os.path.join('output', 'CFT'))
    output_dirs.sort()
    for output_dir in output_dirs:
        path = os.path.join('output', 'CFT', output_dir)
        run_name = os.path.basename(path)
        if run_name not in ['cache'] and os.path.exists(os.path.join(path, 'weight.pt')):
            W = torch.load(os.path.join(path, 'weight.pt')).to(device)
            b = torch.load(os.path.join(path, 'bias.pt')).to(device)

            model.classification_layer.weight = torch.nn.Parameter(W)
            model.classification_layer.bias = torch.nn.Parameter(b)

            preds = torch.zeros((len(valid_dataset), num_categories))
            ys = torch.zeros((len(valid_dataset), num_categories))
            with torch.no_grad():
                enumerater = tqdm(enumerate(valid_dataloader), total=valid_dataloader.__len__())
                for batch, (x, y) in enumerater:
                    x, y = x.to(device), y.to(device)
                    preds[batch*batch_size: (batch+1)*batch_size, :] = model(x)
                    ys[batch*batch_size: (batch+1)*batch_size, :] = y

    # Calculate metrics and logging

            for name, metric in validation_metrics.items():
                result = metric(preds, ys).detach().numpy()
                if len(result.shape) == 0:
                    record = {'Run Name': run_name, 'Result': result}
                    excellog.add('test_'+name, record)
                    print(f'{name}: {result:.4f}')
                else:
                    record = {**{'Run Name': run_name}, **{str(no): result[no] for no in range(len(result))}}
                    excellog.add('test_'+name, record)

            excellog.flush()

if __name__=='__main__':
    main()
    
        