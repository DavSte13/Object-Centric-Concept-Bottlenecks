python generate_concepts.py \
    --dataset voc \
    --data_dir data/voc \
    --encoding_dir encodings

python generate_concepts.py \
    --dataset voc \
    --data_dir data/voc \
    --use_object_concepts \
    --num_objects 10 \
    --min_object_size 0.02 \
    --max_object_size 0.85 \
    --min_score 0.2 \
    --max_iou_threshold 0.5 \
    --object_detection_model mask-rcnn \
    --obj_encoding_dir obj_encodings




