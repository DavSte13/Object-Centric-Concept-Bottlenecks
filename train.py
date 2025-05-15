import torch
torch.set_num_threads(10)
import numpy as np
from model import ConceptBottleneckModelWithEncs
import h5py
from torch.utils.data import DataLoader, TensorDataset
import argparse
import os
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import wandb
import random
from rtpt.rtpt import RTPT
from sklearn.metrics import balanced_accuracy_score, recall_score, average_precision_score

from utils import obj_enc_aggregation, CLASS_NUMBERS, seed_all
from arg_utils import add_general_args, add_object_centric_args, add_training_args, add_wandb_args, add_dataset_args, add_generation_args


def load_splice_embeddings(
    embeddings_path, obj_embeddings_path=None, prediction_level="super", debug=False, num_objects_training=5
):
    """Load the SpLiCE embeddings from the specified path."""
    with h5py.File(embeddings_path, "r") as f:
        train_embeddings = torch.from_numpy(f["train_embeddings"][:])
        train_clip_embeddings = torch.from_numpy(f["train_clip_embeddings"][:])
        train_img_ids = torch.from_numpy(f["train_img_ids"][:])
        test_embeddings = torch.from_numpy(f["test_embeddings"][:])
        test_clip_embeddings = torch.from_numpy(f["test_clip_embeddings"][:])
        test_img_ids = torch.from_numpy(f["test_img_ids"][:])
        if prediction_level == "normal":
            train_labels = torch.from_numpy(f["train_category"][:])
            test_labels = torch.from_numpy(f["test_category"][:])
        elif prediction_level == "super":
            train_labels = torch.from_numpy(f["train_supercategory"][:])
            test_labels = torch.from_numpy(f["test_supercategory"][:])
        else:
            raise ValueError(
                f"Invalid prediction level: {prediction_level}. Choose 'normal' or 'super'."
            )

    # split train in train and val
    train_size = int(0.8 * len(train_embeddings))
    val_size = len(train_embeddings) - train_size

    if obj_embeddings_path is not None:
        # Combine the object embeddings with the train and test embeddings
        with h5py.File(obj_embeddings_path, "r") as f:
            train_obj_embeddings = torch.from_numpy(f["train_obj_embeddings"][:])
            train_num_obj = torch.from_numpy(f["train_num_obj"][:])
            train_obj_img_ids = torch.from_numpy(f["train_obj_img_ids"][:])
            test_obj_embeddings = torch.from_numpy(f["test_obj_embeddings"][:])
            test_num_obj = torch.from_numpy(f["test_num_obj"][:])
            test_obj_img_ids = torch.from_numpy(f["test_obj_img_ids"][:])
        # align image and object embeddings according to their img ids
        # Step 1: Create a mapping from image ID to index in training_obj_img_ids
        train_id_to_index = {img_id.item(): idx for idx, img_id in enumerate(train_obj_img_ids)}
        test_id_to_index = {img_id.item(): idx for idx, img_id in enumerate(test_obj_img_ids)}

        # Step 2: Use that mapping to get the correct indices for reordering
        train_aligned_indices = [train_id_to_index[img_id.item()] for img_id in train_img_ids]
        test_aligned_indices = [test_id_to_index[img_id.item()] for img_id in test_img_ids]

        # Step 3: Convert to a tensor of indices and align the embeddings
        train_aligned_indices_tensor = torch.tensor(train_aligned_indices, dtype=torch.long)
        train_obj_embeddings = train_obj_embeddings[train_aligned_indices_tensor]
        test_aligned_indices_tensor = torch.tensor(test_aligned_indices, dtype=torch.long)
        test_obj_embeddings = test_obj_embeddings[test_aligned_indices_tensor]

        train_obj_embeddings, val_obj_embeddings = torch.split(train_obj_embeddings, [train_size, val_size])
        avg_num_obj = torch.mean(torch.clamp(train_num_obj, min=0, max=num_objects_training).float())
    else:
        N_test, D = test_embeddings.shape
        # Create dummy object embeddings
        train_obj_embeddings = torch.zeros((train_size, 0, D))
        val_obj_embeddings = torch.zeros((val_size, 0, D))
        test_obj_embeddings = torch.zeros((N_test, 0, D))
        avg_num_obj = 0

    # split train and val embeddings (non_obj embeddings)
    train_embeddings, val_embeddings = torch.split(train_embeddings, [train_size, val_size])
    train_clip_embeddings, val_clip_embeddings = torch.split(train_clip_embeddings, [train_size, val_size])
    train_img_ids, val_img_ids = torch.split(train_img_ids, [train_size, val_size])
    train_labels, val_labels = torch.split(train_labels, [train_size, val_size])

    train_dataset = TensorDataset(
        train_embeddings,
        train_clip_embeddings,
        train_obj_embeddings,
        train_img_ids,
        train_labels,
    )
    val_dataset = TensorDataset(
        val_embeddings, val_clip_embeddings, val_obj_embeddings, val_img_ids, val_labels
    )
    test_dataset = TensorDataset(
        test_embeddings,
        test_clip_embeddings,
        test_obj_embeddings,
        test_img_ids,
        test_labels,
    )

    # Store these values for later use
    embedding_dim = train_embeddings.shape[1]
    if debug:
        # change val and test set to have a larger test set for cocologic
        return train_dataset, test_dataset, val_dataset, embedding_dim, avg_num_obj
    else:
        return train_dataset, val_dataset, test_dataset, embedding_dim, avg_num_obj


# --- Training ---
def train_model(
    model,
    train_loader,
    val_loader,
    test_loader,
    dataset,
    use_obj=False,
    aggregation="sum",
    num_epochs=10,
    lr=0.001,
    device="cuda",
    num_objects=5,
    wandb_run=None,
    task_type="multilabel",
):

    model.to(device)
    rtpt = RTPT(
        name_initials=wandb_run.config["initials"],
        experiment_name=wandb_run.name,
        max_iterations=num_epochs,
    )
    rtpt.start()


    # Loss functions
    if task_type == "multiclass":
        if dataset == "cocologic":
            class_counts =  [42, 738, 2955, 5803, 32, 2459, 4902, 2362, 853, 163]
            total = sum(class_counts)
            class_weights = [total / c for c in class_counts]  # Inverse frequency
            class_weights = torch.tensor(class_weights, dtype=torch.float)
            criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
        else:
            criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    # Optimizers
    optimizer_main = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()

        for embeddings, clip_embeddings, obj_embeddings, img_ids, labels in tqdm(train_loader):
            embeddings = embeddings.to(device)
            clip_embeddings = clip_embeddings.to(device)
            if task_type == "multilabel":
                labels = labels.to(device).float()
            else:
                labels = labels.to(device).long()
                # transform back from one-hot to class indices
                labels = torch.argmax(labels, dim=1)
            if use_obj:
                # Aggregate object encodings with the specified method
                obj_embeddings = obj_embeddings[:, :num_objects, :].to(device)
                embeddings = obj_enc_aggregation(aggregation, obj_embeddings, embeddings)

            optimizer_main.zero_grad()

            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_main.step()

            wandb_run.log(
                {
                    "loss/classification": loss.item(),
                }
            )

        print(f"Epoch {epoch+1} finished.")

        # Validation Loop
        model.eval()  # Set model to evaluation mode
        total = 0
        if task_type == "multilabel":
            correct_per_class = torch.zeros(labels.shape[1]).to(args.device)
            collect_labels = []
            collect_preds = []
        else:
            correct = 0

        with torch.no_grad():  # No gradients needed for validation
            for embeddings, clip_embeddings, obj_embeddings, img_ids, labels in tqdm(val_loader):
                embeddings= embeddings.to(device)

                if task_type == "multilabel":
                    labels = labels.to(device).float()
                else:
                    labels = labels.to(device).long()
                    # transform back from one-hot to class indices
                    labels = torch.argmax(labels, dim=1)

                if use_obj:
                    # Aggregate object encodings with the specified method
                    obj_embeddings = obj_embeddings[:, :num_objects, :].to(device)
                    embeddings = obj_enc_aggregation(aggregation, obj_embeddings, embeddings)
                
                outputs = model(embeddings)
                loss = criterion(outputs, labels)
                total += labels.size(0)
                if task_type == "multilabel":
                    # Convert logits to probabilities
                    probabilities = torch.sigmoid(outputs)

                    # Apply threshold (default is 0.5)
                    predicted_classes = (probabilities > 0.5).int()

                    # Track accuracy per class
                    correct_per_class += (predicted_classes == labels).sum(dim=0)
                    collect_labels.append(labels)
                    collect_preds.append(probabilities)
                else:
                    predicted_classes = torch.argmax(outputs, dim=1)  # take the most probable class
                    correct += (predicted_classes == labels).sum()
            
            if task_type == "multilabel":
                # Compute per-class accuracy
                class_accuracies = correct_per_class.float() / total
                val_accuracy = torch.mean(class_accuracies)
                map_score = average_precision_score(torch.cat(collect_labels).cpu(), torch.cat(collect_preds).cpu(), average='macro')
                wandb_run.log({f"val_map": map_score})
            else:
                val_accuracy = correct / total
            wandb_run.log(
                {
                    "val_loss": loss.item(),
                    "val_accuracy": val_accuracy.item(),
                }
            )
            rtpt.step()

    # Testing Loop
    model.eval()
    total = 0
    if task_type == "multilabel":
        correct_per_class = torch.zeros(labels.shape[1]).to(args.device)
        collect_labels = []
        collect_preds = []
    else:
        correct = 0
        collect_labels = []
        preds = []

    with torch.no_grad():
        for embeddings, clip_embeddings, obj_embeddings, img_ids, labels in tqdm(test_loader):
            embeddings = embeddings.to(device)

            if task_type == "multilabel":
                labels = labels.to(device).float()
            else:
                labels = labels.to(device).long()
                # transform back from one-hot to class indices
                labels = torch.argmax(labels, dim=1)

            if use_obj:
                # Aggregate object encodings with the specified method
                obj_embeddings = obj_embeddings[:, :num_objects, :].to(device)
                embeddings = obj_enc_aggregation(aggregation, obj_embeddings, embeddings)
            
            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            total += labels.size(0)

            if task_type == "multilabel":
                # Convert logits to probabilities
                probabilities = torch.sigmoid(outputs)

                # Apply threshold (default is 0.5)
                predicted_classes = (probabilities > 0.5).int()

                # Track accuracy per class
                correct_per_class += (predicted_classes == labels).sum(dim=0)
                collect_labels.append(labels)
                collect_preds.append(probabilities)
            else:
                predicted_classes = torch.argmax(outputs, dim=1)  # take the most probable class
                preds.append(predicted_classes)
                collect_labels.append(labels)

                correct += (predicted_classes == labels).sum()

    if task_type == "multilabel":
        # Compute per-class test accuracy
        test_class_accuracies = correct_per_class.float() / total        
        test_accuracy = torch.mean(test_class_accuracies)
        map_score = average_precision_score(torch.cat(collect_labels).cpu(), torch.cat(collect_preds).cpu(), average='macro')
        wandb_run.log({f"test_map": map_score})

        # Log per-class accuracies
        for i, acc in enumerate(test_class_accuracies):
            wandb_run.log({f"test_accuracy/class_{i}": acc.item()})

        print(f"Per-Class Test Accuracies: {test_class_accuracies}")
    else:
        test_accuracy = correct / total
        balanced_acc = balanced_accuracy_score(torch.cat(collect_labels).cpu(), torch.cat(preds).cpu())
        per_class_acc = recall_score(torch.cat(collect_labels).cpu(), torch.cat(preds).cpu(), average=None)
        print(f"Per-Class Accuracy: {per_class_acc}")
        wandb_run.log({f"test_balanced_accuracy": balanced_acc})
        wandb_run.log({f"test_per_class_accuracy": per_class_acc})

    wandb_run.log({"test_accuracy": test_accuracy.item()})

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CBM Training")
    parser = add_general_args(parser)
    parser = add_object_centric_args(parser)
    parser = add_training_args(parser)
    parser = add_wandb_args(parser)
    parser = add_dataset_args(parser)
    parser = add_generation_args(parser)

    args = parser.parse_args()

    run_name = args.wandb_run_name + f"_{args.prediction_level}_{args.aggregation}_{args.num_objects_training}_size_{args.min_object_size}_score_{args.min_score}"

    run = wandb.init(
        project=args.wandb_project,
        name=run_name,
        config=vars(args),
    )

    seed_all(args.seed)

    base_embeddings_path = os.path.join(args.encoding_dir, args.dataset, f"splice_embeddings_{args.l1_penalty}.h5")
    assert os.path.exists(base_embeddings_path), f"SpLiCE embeddings file {base_embeddings_path} does not exist. Please generate the SpLiCE embeddings first."
    if args.use_object_concepts:
        # check if the concept encodings exist
        file_path = os.path.join(
            args.encoding_dir, args.dataset, args.object_encoding_dir,
            f"obj_{args.num_objects}_min_{args.min_object_size}_max_{args.max_object_size}_score_{args.min_score}_iou_{args.max_iou_threshold}_model_{args.object_detection_model}.h5",
        )
        assert os.path.exists(file_path), f"Object encodings file {file_path} does not exist. Please generate the object encodings first."
        # create datasets with object_encodings
        train_dataset, val_dataset, test_dataset, encoding_dim, avg_num_obj = load_splice_embeddings(
            base_embeddings_path, file_path, prediction_level=args.prediction_level, num_objects_training=args.num_objects_training, debug=args.dataset == "cocologic"
        )
    else:
        # create datasets without object_encodings
        train_dataset, val_dataset, test_dataset, encoding_dim, avg_num_obj = load_splice_embeddings(
            base_embeddings_path, prediction_level=args.prediction_level, debug=args.dataset == "cocologic"
        )


    # reproducible data loading
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(args.seed)

    # log the average number of object proposals in the dataset
    run.log({"avg_num_obj": avg_num_obj})
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g, num_workers=1, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, worker_init_fn=seed_worker, generator=g, num_workers=1, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, worker_init_fn=seed_worker, generator=g, num_workers=1, pin_memory=False)
    # Create log directory if it doesn't exist
    os.makedirs(args.log_dir, exist_ok=True)

    if args.use_object_concepts and args.aggregation == "concat":
        # if we concatenate encodings, the encoding dimension is larger
        encoding_dim = encoding_dim * (args.num_objects_training + 1)
    elif args.use_object_concepts and args.aggregation == "sum_count":
        # the combination of sum with object counts has a different encoding dimension
        encoding_dim = encoding_dim * 2

    num_classes = CLASS_NUMBERS[args.dataset][0] if args.prediction_level == "super" else CLASS_NUMBERS[args.dataset][1]


    model = ConceptBottleneckModelWithEncs(
        num_classes=num_classes,
        explicit_encoding_dim=encoding_dim,
        device=args.device,
    )

    train_model(
        model,
        train_loader,
        val_loader,
        test_loader,
        dataset=args.dataset,
        use_obj=args.use_object_concepts,
        aggregation=args.aggregation,
        num_epochs=args.num_epochs,
        lr=args.learning_rate,
        device=args.device,
        num_objects=args.num_objects_training,
        wandb_run=run,
        task_type=args.task_type,
    )
