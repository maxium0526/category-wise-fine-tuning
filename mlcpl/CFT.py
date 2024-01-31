import torch
import numpy as np
import os
from datetime import datetime
from pathlib import Path
from .label_strategy import *
from .helper import *
from torch.utils.data import Dataset, DataLoader

class FTDataset(Dataset):
    def __init__(self, Z, Y):
        self.Z = Z
        self.Y = Y
        self.num_categories = self.Y.shape[1]

    def __len__(self):
        return self.Z.shape[0]

    def __getitem__(self, idx):
        z, y = self.Z[idx], self.Y[idx]
        return z, y
        
    def test(self):
        return self.__getitem__(0)

def CFT(
    weight,
    bias,
    training_data=None,
    validation_data=None,
    optimizer=None,
    batch_size=None,
    epochs=1,
    early_stopping=None,
    validation_metric=None,unknown_label_strategy=unknown_to_unknown,
    uncertain_label_strategy=uncertain_to_unknown,
    device='cuda',
    tblog=None,
    excellog=None,
    ):

    num_categories = weight.shape[0]

    z_train, y_train = training_data
    z_valid, y_valid = validation_data
    z_train, y_train = z_train.to(device), unknown_label_strategy(uncertain_label_strategy(y_train.to(device)))
    z_valid, y_valid = z_valid.to(device), y_valid.to(device)
    
    weight, bias = weight.to(device), bias.to(device)

    finetuned_weight = weight.clone().detach()
    finetuned_bias = bias.clone().detach()

    for i in range(num_categories):
        # prepare head
        head_weight = torch.nn.Parameter(weight[i:i+1, :].clone().detach())
        head_bias = torch.nn.Parameter(bias[i:i+1].clone().detach())

        # prepare training data
        head_z_train, head_y_train = z_train, y_train[:, i:i+1]
        train_label_map = ~torch.isnan(head_y_train).view(-1)
        head_z_train, head_y_train = head_z_train[train_label_map, :], head_y_train[train_label_map, :]
        # head_z_train, head_y_train = head_z_train.to(config.DEVICE), head_y_train.to(config.DEVICE)

        # prepare validation data
        head_z_valid, head_y_valid = z_valid, y_valid[:, i:i+1]
        valid_label_map = ~torch.isnan(head_y_valid).view(-1)
        head_z_valid, head_y_valid = head_z_valid[valid_label_map, :], head_y_valid[valid_label_map, :]
        # head_z_valid, head_y_valid = head_z_valid.to(config.DEVICE), head_y_valid.to(config.DEVICE)

        if (head_y_train==0).sum() == 0 or (head_y_train==1).sum() == 0:
            print(f'Category {i} cannot be trained. Skip.')
            continue
        if (head_y_valid==0).sum() == 0 or (head_y_valid==1).sum() == 0:
            print(f'Category {i} cannot be validated. Skip.')
            continue
        
        print(f'Fine-tuning category {i}/{num_categories}.')

        head_best_weight, head_best_bias, records = finetune_head(
            head_weight,
            head_bias,
            training_data=(head_z_train, head_y_train),
            validation_data=(head_z_valid, head_y_valid),
            optimizer=optimizer,
            batch_size=batch_size,
            epochs=epochs,
            early_stopping=early_stopping,
            validation_metric=validation_metric,
            device=device,
        )

        finetuned_weight[i:i+1, :] = head_best_weight
        finetuned_bias[i:i+1] = head_best_bias

        # writing logs to loggers
        for record in records:
            excellog.add('category_'+str(i), record)
            tblog.add_scalars('category_'+str(i), record, record['Epoch'])
    excellog.flush()
    tblog.flush()

    return finetuned_weight, finetuned_bias

def finetune_head(
    weight,
    bias,
    training_data=None,
    validation_data=None,
    optimizer=None,
    batch_size=None,
    epochs = 1,
    early_stopping=None,
    validation_metric=None,
    device='cuda',
    ):

    z_train, y_train = training_data
    z_valid, y_valid = validation_data
    z_train, y_train = z_train.to(device), y_train.to(device)
    z_valid, y_valid = z_valid.to(device), y_valid.to(device)
    
    head = torch.nn.Linear(z_train.shape[1], 1).to(device)
    head.weight = torch.nn.Parameter(weight.clone().detach())
    head.bias = torch.nn.Parameter(bias.clone().detach())

    optimizer.set_head(head)

    original_validation_score = validation_metric(head(z_valid), y_valid)
    print(f'Original Vaild Score: {original_validation_score:.4f}')

    best_validation_score = original_validation_score
    best_weight, best_bias = head.weight.clone().detach(), head.bias.clone().detach()
    best_at = -1

    records = []

    for epoch in range(epochs):
        record = {'Epoch': epoch}

        train_log = optimizer.step(z_train, y_train)
        train_log = {name: value.cpu().detach().numpy() for name, value in train_log.items()}
        
        with torch.no_grad():
            validation_score = validation_metric(head(z_valid), y_valid)
        
        record.update(train_log)
        record['Valid Score'] = validation_score.cpu().detach().numpy()


        if validation_score > best_validation_score:
            best_validation_score = validation_score
            best_at = epoch
            best_weight, best_bias = head.weight.clone().detach(), head.bias.clone().detach()

        records.append(record)
        print_record(record)

        if early_stopping is not None and epoch - best_at >= early_stopping:
            print()
            print(f'Early stopping. Best Valid Score: {best_validation_score:.4f} (+{best_validation_score-original_validation_score:.4f})')
            print()
            break

        if epoch == epochs - 1:
            print()
            print(f'Done. Best Valid Score: {best_validation_score:.4f} (+{best_validation_score-original_validation_score:.4f})')
            print()
            break
    
    del z_train, z_valid, y_train, y_valid
    torch.cuda.empty_cache()

    return best_weight, best_bias, records

class HeuristicOptimizer():
    def __init__(self):
        pass
    
    def set_head(self, head):
        self.head = head

    def step(self, x, y):
        pass

    @staticmethod
    def encode(weight, bias):
        return torch.concatenate([weight.reshape(-1), bias.reshape(-1)], axis=0)
    
    @staticmethod
    def decode(sol):
        weight = sol[:-1].reshape(1, -1)
        bias = sol[-1].reshape(-1)
        return weight, bias

    @staticmethod
    def batch_fitness(population, x, y, metric):

        weight = population[:, :-1]
        bias = population[:, -1]
        preds = torch.nn.functional.linear(x, weight, bias=bias).transpose(0, 1).unsqueeze(-1)

        fitnesses = torch.zeros(preds.shape[0])
        for i in range(fitnesses.shape[0]):
            fitnesses[i] = metric(preds[i], y)

        return fitnesses

class GAOptimizer(HeuristicOptimizer):
    def __init__(
        self,
        head = None,
        device = 'cuda',
        metric = None,
        higher_is_better = True,
        init_pop_var = 0.0,
        num_pop = 50,
        Cr = 0.2,
        mutation_p = 0.5,
        mutation_percent_genes = 1,
        mutation_range = 0.001,
        elitism = 1,

        ):
        self.metric = metric
        self.device = device

        self.higher_is_better = higher_is_better
        self.init_pop_var = init_pop_var
        self.num_pop = num_pop
        self.Cr = Cr
        self.mutation_p = mutation_p
        self.mutation_percent_genes = mutation_percent_genes
        self.mutation_range = mutation_range
        self.elitism = elitism

        if head:
            self.set_head(head)

    def set_head(self, head):
        self.head = head
        self.population = torch.stack([self.encode(head.weight.clone().detach(), head.bias.clone().detach())] * self.num_pop).to(self.device)
        self.step_count = 0

    def step(self, x, y):
        with torch.no_grad():
            self.step_count += 1
            if (y==0.0).sum() == 0 or (y==1.0).sum() == 0:
                return

            pop_fitnesses = self.batch_fitness(self.population, x, y, self.metric)

            if self.higher_is_better:
                best_index = torch.argmax(pop_fitnesses)
                best_pop = self.population[best_index]
                best_fitness = torch.max(pop_fitnesses)
                avg_fitness = torch.mean(pop_fitnesses)
                parents_1_indices = torch.multinomial(pop_fitnesses, self.population.shape[0], replacement=True)
                parents_2_indices = torch.multinomial(pop_fitnesses, self.population.shape[0], replacement=True)
            else:
                best_index = torch.argmin(pop_fitnesses)
                best_pop = self.population[best_index]
                best_fitness = torch.min(pop_fitnesses)
                avg_fitness = torch.mean(pop_fitnesses)

                parents_1_indices = torch.multinomial(-pop_fitnesses, self.population.shape[0], replacement=True)
                parents_2_indices = torch.multinomial(-pop_fitnesses, self.population.shape[0], replacement=True)

            weight, bias = self.decode(best_pop)
            self.head.weight, self.bias = torch.nn.Parameter(weight), torch.nn.Parameter(bias)

            parents_1 = self.population[parents_1_indices]
            parents_2 = self.population[parents_2_indices]
            offsprings = torch.where(torch.rand_like((self.population)) < self.Cr, parents_1, parents_2)

            mutations = (torch.rand_like(offsprings) * 2 - 1) * self.mutation_range
            mutations = torch.where(torch.rand_like(mutations) < self.mutation_percent_genes, mutations, torch.zeros_like(mutations))
            mutations = torch.where((torch.rand_like(mutations)[:, 0] < self.mutation_p)[..., None], mutations, torch.zeros_like(mutations))
            offsprings = offsprings + mutations.to(self.device)

            if self.elitism > 0:
                sorted_population = self.population[torch.sort(pop_fitnesses, descending=self.higher_is_better)[1]]
                offsprings = torch.concat([sorted_population[: self.elitism], offsprings[self.elitism:]])

            self.population = offsprings

            log = {
                'best_fitness': best_fitness,
                'avg_fitness': avg_fitness,
            }

            return log

class BPOptimizer():
    def __init__(
        self, 
        head = None,
        device = 'cuda',
        loss_fn = None,
        optimizer_class = None,
        optimizer_kwargs = None,
        ):
        self.loss_fn = loss_fn
        self.device = device
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs

        if head:
            self.set_head(head)

    def set_head(self, head):
        self.head = head
        self.step_count = 0
        self.optimizer = self.optimizer_class(self.head.parameters(), **self.optimizer_kwargs)

    def step(self, x, y):
        self.step_count += 1
        if (y==0.0).sum() == 0 or (y==1.0).sum() == 0:
            return

        pred = self.head(x)
        loss = self.loss_fn(pred, y)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        log = {
            'loss': loss,
        }

        return log