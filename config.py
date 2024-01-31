from mlcpl import dataset

train_dataset = dataset.MSCOCO('/home/max/datasets/COCO', split='train', partial_ratio=0.1)
valid_dataset = dataset.MSCOCO('/home/max/datasets/COCO', split='valid')
