# Object-Centric-Concept-Bottlenecks

This repository contains code for computing concept embeddings and training predictor models using object-centric bottlenecks. The approach extends traditional concept-based models by incorporating object-level semantics for improved interpretability and performance.

---

## Installation

You can set up the environment in one of two ways:

### Option 1: Docker (Recommended)

Use the provided Docker environment to ensure full compatibility.  
Instructions for building and running the container can be found in the [`.docker/`](.docker/) directory.

### Option 2: Manual Environment

If you prefer setting up manually, create a Python environment (Python 3.8+ recommended), and install required packages. In addition to the usual dependencies, **you must install the following manually**:

- SpLiCE: https://github.com/AI4LIFE-GROUP/SpLiCE
- SAM (Segment Anything Model): ``pip install git+https://github.com/facebookresearch/segment-anything.git``

---

## Embedding Computation

To compute the concept embeddings, run:

```bash
python generate_concepts.py
--dataset                Dataset to use ['coco', 'sun', 'voc', 'cocologic']
--data_dir               Path to the image dataset
--encoding_dir           Where to store image-level embeddings (default: encodings)
```

To enable object-based encodings, use the flag ``--use_object_concepts``. Additional relevant options include:

```
--num_objects              Number of objects per image (default: 10)
--min_object_size          Minimum object size (relative, default: 0.02)
--max_object_size          Maximum object size (relative, default: 0.85)
--min_score                Minimum object proposal score (default: 0.2)
--max_iou_threshold        Max IoU threshold for filtering proposals (default: 0.5)
--object_encoding_dir      Output directory for object encodings (default: obj_encodings)
--object_detection_model   Detection model to use ['mask-rcnn', 'sam']
```

## Model Training
Once embeddings are generated, you can train a predictor model via `` python train.py ``
### Basic Training Settings:
```
--num_epochs              Number of training epochs (default: 40)
--learning_rate           Learning rate (default: 0.001)
--prediction_level        Class granularity ['super', 'normal']
--task_type               Task type ['multilabel', 'multiclass']
--dataset                 Dataset name ['coco', 'sun', 'voc', 'cocologic', 'corel']
--encoding_dir            Path to image-level embeddings
```

### Additional Parameters for OCB Training:

```
--aggregation             Object encoding aggregation ['sum', 'max', 'concat', sum_count', 'count']
--num_objects              Same as during embedding
--object_encoding_dir      "
--min_object_size          "
--max_object_size          "
--min_score                "
--max_iou_threshold        "
--object_detection_model   "
```

### Scripts

All helper scripts for precomputing embeddings and launching training can be found in the ``scripts/`` directory.