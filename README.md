# Category-wise Fine-Tuning

- [Conference Paper](https://link.springer.com/chapter/10.1007/978-981-99-8145-8_26)
- [Journal Preprint](http://arxiv.org/abs/2401.16991)

## Introduction
This code demonstrates using CFT_BP_ASL and CFT_GA to fine-tune models trained with Assume Negative (AN) (i.e., Negative mode) on the MS-COCO dataset.

1. Download [the MS-COCO 2014 dataset](https://cocodataset.org/#download).
2. Put the training image folder (`train2014`), validation image folder (`val2014`), and annotation folder (`annotations`) under the same folder (eg., `COCO`) like this:
   
        -----COCO  
           |----train2014  
           |----val2014  
           |----annotations
3. Run `pip install -r requirements.py` to install the necessary packages.
3. Config the dataset location in `config.py`.
4. Config the known label proportion in `config.py`.
5. Run `python train.py`. This will train a classification model with AN and save the trained model in `output/train/best.pth`.
6. Run `CFT_prepare.py`. This will generate and store (feature vector z, label y) pairs of the dataset to `output/CFT/cache` for preparation of CFT. This generation can dramatically speed up CFT.
7. Run `CFT_optimize.py`. This will respectively use CFT_BP_ASL and CFT_GA to fine-tune the train model. The parameters of the fine-tuned classification layer will be saved in `output/CFT`.
8. Run `validate.py` to see the classification performances of the trained model, the trained model after CFT_BP_ASL, and the trained model after CFT_GA. The result is saved in `output/valid/logs`.

## CheXpert Competition Single Model

See this repo: [maxium0526/cft-chexpert](https://github.com/maxium0526/cft-chexpert).

## Acknowledgment

Part of the codes in this repository are from:

- [Alibaba-MIIL/ASL](https://github.com/Alibaba-MIIL/ASL)
- [Alibaba-MIIL/PartialLabelingCSL](https://github.com/Alibaba-MIIL/PartialLabelingCSL)
