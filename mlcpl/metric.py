import torch
import numpy as np

class PartialMultilabelMetric():
    def __init__(self, binary_metric, mask=None, reduction='mean'):
        self.binary_metric = binary_metric
        self.mask = mask
        self.reduction = reduction
        
    def __call__(self, preds, target, threshold=None):
        with torch.no_grad():
            num_categories = preds.shape[1]
            scores = torch.zeros(num_categories, dtype=torch.float32)

            for i in range(num_categories):
                category_preds, category_target = preds[:, i], target[:, i]
                scores[i] = PartialBinaryMetric(self.binary_metric)(category_preds, category_target, threshold=threshold)
            
            if self.reduction is None:
                return scores
            if self.reduction=='mean':
                if self.mask is None:
                    mean_score = torch.mean(scores[~torch.isnan(scores)])
                else:
                    selected_scores = scores[self.mask]
                    mean_score = torch.mean(selected_scores[~torch.isnan(selected_scores)])

                return mean_score

class PartialBinaryMetric():
    def __init__(self, binary_metric):
        self.binary_metric = binary_metric

    def __call__(self, preds, target, threshold=None):
        with torch.no_grad():
            labeled_map = ~torch.isnan(target)

            preds, target = preds[labeled_map], target[labeled_map]

            if target.numel() == 0 :
                score = torch.tensor(torch.nan)
            elif (target==1.0).sum() == 0:
                score = torch.tensor(torch.nan)
            elif (target==0.0).sum() == 0:
                score = torch.tensor(torch.nan)
            else:
                if threshold:
                    score = self.binary_metric(preds, target.to(torch.int32), threshold=threshold)
                else:
                    score = self.binary_metric(preds, target.to(torch.int32))

            return score