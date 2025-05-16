# Hyperparameter settings for training the predictor network

## Baselines

COCO: LR 0.001, Num Epochs: 20

COCOLogic: LR 0.001, Num Epochs: 40

SUN: LR 0.001, Num Epochs: 40

VOC: LR 0.001, Num Epochs: 30


## OCB

### concat, max, sum

COCO: 
- Mask-RCNN: LR 0.001, Num Epochs 20
- SAM: LR 0.001, Num Epochs 60

COCOLogic:
- Mask-RCNN: LR 0.001, Num Epochs 40
- SAM: LR 0.001, Num Epochs 100

VOC:
- Mask-RCNN: LR 0.001, Num Epochs 30
- SAM: LR 0.001, Num Epochs 20

SUN:
- Mask-RCNN: LR 0.001, Num Epochs 60
- SAM: LR 0.001, Num Epochs 60

### count, sum + count
COCO: 
- Mask-RCNN: LR 0.0001, Num Epochs 20
- SAM: LR 0.0001, Num Epochs 60

COCOLogic:
- Mask-RCNN: LR 0.0001, Num Epochs 40
- SAM: LR 0.0001, Num Epochs 100

VOC:
- Mask-RCNN: LR 0.0001, Num Epochs 30
- SAM: LR 0.0001, Num Epochs 20

SUN:
- Mask-RCNN: LR 0.0001, Num Epochs 60
- SAM: LR 0.0001, Num Epochs 60
