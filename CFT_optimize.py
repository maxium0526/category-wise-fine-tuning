import torch
from torch.utils.tensorboard import SummaryWriter
import torchmetrics
import os
from datetime import datetime
from pathlib import Path
from mlcpl.CFT import *
from mlcpl.helper import *
from mlcpl.loss import AsymmetricLoss
from models import *

optimizers = {
    'BP-Asym': BPOptimizer(loss_fn=AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05), optimizer_class=torch.optim.Adam, optimizer_kwargs={'lr': 1e-4}),
    'GA': GAOptimizer(metric=torchmetrics.functional.classification.binary_average_precision),
}

z_train = torch.load(os.path.join('output', 'CFT', 'cache', 'z_train.pt')).cpu()
y_train = torch.load(os.path.join('output', 'CFT', 'cache', 'y_train.pt')).cpu()
z_valid = torch.load(os.path.join('output', 'CFT', 'cache', 'z_valid.pt')).cpu()
y_valid = torch.load(os.path.join('output', 'CFT', 'cache', 'y_valid.pt')).cpu()
num_categories = y_valid.shape[1]

original_weight = torch.load(os.path.join('output', 'CFT', 'original', 'weight.pt')).to('cuda')
original_bias = torch.load(os.path.join('output', 'CFT', 'original', 'bias.pt')).to('cuda')

for name, optimizer in optimizers.items():

    output_dir = os.path.join('output', 'CFT', name+'_'+datetime.now().strftime('%Y%m%d%H%M%S'))
    log_dir = os.path.join(output_dir, 'log')
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    tblog = SummaryWriter(log_dir=log_dir)
    excellog = ExcelLogger(os.path.join(log_dir, 'excel_log.xlsx'))

    finetuned_weight, finetuned_bias = CFT(
        original_weight,
        original_bias,
        training_data=(z_train, y_train),
        validation_data=(z_valid, y_valid),
        validation_metric=torchmetrics.functional.classification.binary_average_precision,
        optimizer=optimizer,
        epochs=5000,
        early_stopping=300,
        tblog=tblog,
        excellog=excellog,
        )

    torch.save(finetuned_weight, os.path.join(output_dir, 'weight.pt'))
    torch.save(finetuned_bias, os.path.join(output_dir, 'bias.pt'))