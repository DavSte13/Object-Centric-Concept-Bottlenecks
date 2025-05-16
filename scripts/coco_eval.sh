#!/bin/bash

python train.py \
    --dataset coco \
    --task_type multilabel \
    --num_epochs 20 \
    --use_object_concepts \
    --min_object_size 0.01 \
    --max_object_size 0.85 \
    --learning_rate 0.01 \
    --min_score 0.2 \
    --aggregation sum \
    --num_objects_training 5 \
    --seed 42 \
    --prediction_level super \
    --object_detection_model mask-rcnn
