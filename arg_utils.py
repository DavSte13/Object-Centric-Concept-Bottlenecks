import argparse


def add_general_args(parser: argparse.ArgumentParser):
    parser.add_argument_group("General settings")
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers for data loading"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for training (cuda or cpu)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training or generation",
    )
    parser.add_argument(
        "--initials", type=str, default="DS", help="Initials to display via rtpt"
    )
    return parser


def add_dataset_args(parser: argparse.ArgumentParser):
    parser.add_argument_group("Dataset settings")
    parser.add_argument(
        "--dataset",
        type=str,
        default="sun",
        help="Dataset for training or generation (coco, sun, voc, cocologic, cifar)",
        choices=["coco", "sun", "voc", "cocologic", "cifar"],
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/sun",
        help="Path to the directory containing the images",
    )
    parser.add_argument(
        "--encoding_dir",
        type=str,
        default="encodings",
        help="Path to the directory where the concepts will be stored",
    )
    parser.add_argument(
        "--object_encoding_dir",
        type=str,
        default="obj_encodings",
        help="Path to the object encodings",
    )
    return parser


def add_generation_args(parser: argparse.ArgumentParser):
    parser.add_argument_group("Embedding generation settings")
    parser.add_argument(
        "--model",
        type=str,
        default="clip",
        choices=["clip"],
        help="Base model for Splice",
    )
    parser.add_argument(
        "--vocabulary", type=str, default="laion", help="Vocabulary for Splice"
    )
    parser.add_argument(
        "--size", type=int, default=10000, help="Size of the vocabulary"
    )
    parser.add_argument(
        "--l1_penalty", type=float, default=0.25, help="L1 penalty for Splice"
    )
    parser.add_argument(
        "--apply_sparsity",
        type=str,
        default="true",
        choices=["true", "false"],
        help="Apply sparsity of the Splice model",
    )
    return parser


def add_object_centric_args(parser: argparse.ArgumentParser):
    parser.add_argument_group("Object centric generation/training settings")
    parser.add_argument(
        "--use_object_concepts",
        default=False,
        action="store_true",
        help="Train with object concepts",
    )
    parser.add_argument(
        "--num_objects",
        type=int,
        default=10,
        help="Number of objects to use for object-centric concept generation.",
    )
    parser.add_argument(
        "--min_object_size",
        type=float,
        default=0.02,
        help="Minimum size of objects for object proposals",
    )
    parser.add_argument(
        "--max_object_size",
        type=float,
        default=0.85,
        help="Maximum size of objects for object proposals",
    )
    parser.add_argument(
        "--min_score",
        type=float,
        default=0.2,
        help="Minimum score for object proposals",
    )
    parser.add_argument(
        "--max_iou_threshold",
        type=float,
        default=0.5,
        help="Maximum IoU threshold for object proposals",
    )
    parser.add_argument(
        "--draw_bbox",
        default=False,
        action="store_true",
        help="If the object proposals should be drawn on the image for CLIP processing (by default the object is cropped)",
    )
    parser.add_argument(
        "--only_objects",
        default=False,
        action="store_true",
        help="If True, only aggregate object encodings, otherwise combine with image encodings",
    )
    parser.add_argument(
        "--aggregation",
        type=str,
        default="sum",
        help="Aggregation method for object encodings (sum, max, concat, sum_count, count)",
        choices=["sum", "max", "concat", "sum_count", "count"],
    )
    parser.add_argument(
        "--object_detection_model",
        type=str,
        default="mask-rcnn",
        help="Object detection model to use (mask-rcnn, sam)",
        choices=["mask-rcnn", "sam"],
    )
    parser.add_argument(
        "--no_concepts",
        default=False,
        action="store_true",
        help="If True, do not use concepts in the model",
    )
    return parser


def add_training_args(parser: argparse.ArgumentParser):
    parser.add_argument_group("Training settings")
    parser.add_argument(
        "--log_dir", type=str, default="logs", help="Path to the log directory"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=10, help="Number of epochs for training"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Learning rate for training"
    )
    parser.add_argument(
        "--prediction_level",
        type=str,
        default="super",
        help="Multilabel classification for supercategories or (normal) categories.",
        choices=["super", "normal"],
    )
    parser.add_argument(
        "--num_objects_training",
        type=int,
        default=5,
        help="Number of objects (and their concepts) to use for training.",
    )
    parser.add_argument(
        "--task_type",
        type=str,
        default="multilabel",
        help="Task type for training (multilabel or multiclass)",
        choices=["multilabel", "multiclass"],
    )
    return parser


def add_wandb_args(parser: argparse.ArgumentParser):
    parser.add_argument_group("WandB settings")
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="OCB",
        help="WandB project name",
    )
    parser.add_argument(
        "--wandb_run_name", type=str, default="OCB_eval", help="WandB run name"
    )
    return parser
