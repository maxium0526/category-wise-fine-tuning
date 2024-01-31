import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import transforms
import torchmetrics
import os
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from mlcpl.loss import AsymmetricLoss
from mlcpl.helper import ExcelLogger
from mlcpl.label_strategy import *
from mlcpl.metric import PartialMultilabelMetric
from models import Model
import config

# from https://github.com/Alibaba-MIIL/PartialLabelingCSL/blob/main/src/helper_functions/helper_functions.py
def add_weight_decay(model, weight_decay=1e-4, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]

# from https://github.com/Alibaba-MIIL/PartialLabelingCSL/blob/main/src/helper_functions/helper_functions.py
class ModelEma(torch.nn.Module):
    def __init__(self, model, decay=0.9997, device=None):
        super(ModelEma, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)

def main():
    device = 'cuda'
    output_dir = 'output/train'

    train_transform = transforms.Compose([
        transforms.RandAugment(interpolation=torchvision.transforms.functional.InterpolationMode.BICUBIC),
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
    ])

    valid_transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
    ])

    train_dataset = config.train_dataset
    train_dataset.transform = train_transform

    train_dataset.df.to_csv('train.csv')

    valid_dataset = config.valid_dataset
    valid_dataset.transform = valid_transform

    valid_dataset.df.to_csv('valid.csv')

    num_categories = train_dataset.num_categories

    model = Model(num_categories)
    weight_decay = 1e-4

    loss_fn = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05)

    batch_size = 32
    accum_step = 4
    lr = 2e-4
    epochs = 60
    early_stopping = 10

    training_metrics = {}
    validation_metrics = {
        'mAP': PartialMultilabelMetric(torchmetrics.functional.classification.binary_average_precision),
    }
    monitor_validation_metric_name = 'mAP'
    monitor_mode = 'max'

    ### below normally fixed
    log_dir = os.path.join(output_dir, 'log')
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=20, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=20, shuffle=False)
    
    model = model.to(device)
    parameters = add_weight_decay(model, weight_decay=weight_decay)

    ema = ModelEma(model, 0.999)

    optimizer = torch.optim.Adam(parameters, lr=lr, weight_decay=0)
    steps_per_epoch = len(train_dataloader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, steps_per_epoch=steps_per_epoch, epochs=epochs, pct_start=0.2)

    tblog = SummaryWriter(log_dir=log_dir)
    excellog = ExcelLogger(os.path.join(log_dir, 'excel_log.xlsx'))

    train_dataset.df.to_csv(os.path.join(log_dir, 'training_dataset.csv'))

    best_score = 0
    best_at_epoch = -1

    try:
        for epoch in range(epochs):
            print(f'Epoch: {epoch}/{epochs}')

            # Train Loop
            model.train()

            losses, preds, targets = [], [], []
            bar = tqdm(enumerate(train_dataloader), total=train_dataloader.__len__())
            for batch, (x, y) in bar:
                x, y = x.to(device), unknown_to_negative(y.to(device))
                pred = model(x)
                loss = loss_fn(pred, y)
                loss = loss / accum_step
                loss.backward()

                if (batch + 1) % accum_step == 0 or (batch + 1) == len(train_dataloader):
                    optimizer.step()
                    scheduler.step()
                    ema.update(model)
                    model.zero_grad()

                bar.set_postfix({'Loss': f'{loss.cpu().detach().numpy():.4f}'})
                losses.append(loss.detach().cpu().clone())
                preds.append(pred.detach().cpu().clone())
                targets.append(y.detach().cpu().clone())
            losses = torch.tensor(losses)
            preds = torch.cat(preds)
            targets = torch.cat(targets)

            # Calculate metrics and logging
            mean_loss = torch.mean(losses).detach().numpy()
            tblog.add_scalar('loss', mean_loss, epoch)
            excellog.add('loss', mean_loss)

            for name, metric in training_metrics.items():
                result = metric(preds, targets).detach().numpy()
                if len(result.shape) == 0:
                    tblog.add_scalar('train_'+name, result, epoch)
                    excellog.add('train_'+name, result)
                else:
                    tblog.add_scalars('train_'+name, {str(no): result[no] for no in range(num_categories)}, epoch)
                    excellog.add('train_'+name, {str(no): result[no] for no in range(num_categories)})

            # Valid Loop
            model.eval() 
            preds, ys = [], []
            with torch.no_grad():
                enumerater = tqdm(enumerate(valid_dataloader), total=valid_dataloader.__len__())
                for batch, (x, y) in enumerater:
                    x, y = x.to(device), y.to(device)
                    preds.append(ema.module(x))
                    ys.append(y)
            preds, ys = torch.cat(preds), torch.cat(ys)

            # Calculate metrics and logging
            for name, metric in validation_metrics.items():
                result = metric(preds, ys).detach().numpy()
                if len(result.shape) == 0:
                    tblog.add_scalar('valid_'+name, result, epoch)
                    excellog.add('valid_'+name, result)
                    print(f'{name}: {result:.4f}')
                else:
                    tblog.add_scalars('valid_'+name, {str(no): result[no] for no in range(num_categories)}, epoch)
                    excellog.add('valid_'+name, {str(no): result[no] for no in range(num_categories)})
            
                if name == monitor_validation_metric_name:
                    current_score = result


            if (monitor_mode == 'max' and current_score > best_score) or ((monitor_mode == 'min' and current_score < best_score)):
                best_score = current_score
                best_at_epoch = epoch
                print(f'New best {monitor_validation_metric_name}: {best_score:.4f}')
                torch.save(ema.module.state_dict(), os.path.join(output_dir, 'best.pth'))

            if early_stopping is not None:
                if epoch - best_at_epoch >= early_stopping:
                    print('Early stopping.')
                    break

    except KeyboardInterrupt:
        tblog.flush()
        excellog.flush()
    tblog.flush()
    excellog.flush()

if __name__=='__main__':
    main()
    
        