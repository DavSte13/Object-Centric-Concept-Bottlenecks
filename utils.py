import numpy as np
from PIL import ImageDraw
import torch
from torchvision.models.detection.mask_rcnn import MaskRCNN
import torchvision.transforms.functional as TF
import os
from torch.utils.data import Subset
from segment_anything import SamAutomaticMaskGenerator
from segment_anything.modeling.sam import Sam

CLASS_NUMBERS = {
    "coco": [12, 90],
    "sun": [397, 397],
    "voc": [20, 20],
    "cocologic": [10, 10],
    "cifar": [100, 100],
}


def bbox_to_xyxy(box):
    """
    Convert bounding box from xywh format to xyxy format.
    Args:
        box (list): Bounding box in xywh format [x, y, width, height].
    Returns:
        list: Bounding box in xyxy format [x1, y1, x2, y2].
    """
    x1 = box[0]
    y1 = box[1]
    x2 = box[0] + box[2]
    y2 = box[1] + box[3]
    return [x1, y1, x2, y2]


def draw_box_on_image(image_tensor, box, color=(255, 0, 0), width=2):
    """
    Draw a single bounding box on a copy of the image_tensor.
    image_tensor: Tensor of shape (3, H, W)
    box: Tensor of shape (4,) with coordinates (x1, y1, x2, y2)
    Returns a PIL image with the box drawn.
    """
    image_pil = TF.to_pil_image(image_tensor.cpu())
    draw = ImageDraw.Draw(image_pil)
    x1, y1, x2, y2 = box.long()
    draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
    return image_pil


def gather_object_embeddings(
    boxes,
    scores,
    min_score,
    num_objects,
    image,
    device,
    splice_model,
    embedding_dim,
    transform,
    draw_boxes=False,
):
    """
    Instead of cropping to boxes, this function draws bounding boxes on the image and encodes it.
    Returns: (obj_embeddings: Tensor[num_objects, embedding_dim], num_boxes_used: int)
    """
    if len(boxes) > 0:
        boxes = boxes[scores > min_score]

    if len(boxes) > num_objects:
        boxes = boxes[:num_objects]

    processed_images = []
    for box in boxes:
        if draw_boxes:
            # Draw bounding box on full image
            # In principle, the processor in this case does not need to resize the image, but we do it to ensure consistency
            boxed_pil = draw_box_on_image(image, box)
            processed_image = TF.to_tensor(boxed_pil)  # (3, H, W)
        else:
            # Crop image to bounding box
            x1, y1, x2, y2 = box.long()
            processed_image = image[:, y1:y2, x1:x2]
        processed = transform(processed_image.unsqueeze(0)).squeeze(0)
        processed_images.append(processed)

    if len(processed_images) > 0:
        boxed_images = torch.stack(processed_images).to(device)
        obj_embeddings, _ = splice_model.encode_image(boxed_images)
    else:
        obj_embeddings = torch.zeros(1, embedding_dim, device=device)

    return obj_embeddings, len(boxes)


def generate_object_propsals(
    images, model, min_size=0.05, max_size=0.8, iou_threshold=0.5, minimum_score=0.5
):
    """
    Generate object proposals for a given image. These should be bounding boxes/image crops that contain objects of the given image.
    We want to filter out object proposals that are too large (cover most of the image) or too small (do not cover enough of the image),
    as the former most likely have similar splice concepts than the whole image and splice most likely does not provide good concepts for the latter.

    We use MaskRCNN (pretrained from torchvision) or SAM to generate object proposals.

    Args:
    images: batch of images (tensors)
    model: MaskRCNN model or SAM model (it is assumed that the model and images are on the same device)
    min_size: minimum size of a bounding box (relative to image size)
    max_size: maximum size of a bounding box (relative to image size)
    iou_threshold: maximum iou between two bounding boxes to be considered as different
    minimum_score: minimum certainty score to consider a bounding box

    Returns:
    List of tuples, where each tuple contains the bounding boxes and the corresponding scores for an image.
    """

    # check whether the model is a MaskRCNN model
    if isinstance(model, MaskRCNN):
        # compute the segmentation mask
        with torch.no_grad():
            outputs = model(images)

    elif isinstance(model, Sam):
        # compute the segmentation mask
        with torch.no_grad():

            mask_generator = SamAutomaticMaskGenerator(
                model=model,
                points_per_side=16,  # control density
                pred_iou_thresh=0.92,  # high quality
                stability_score_thresh=0.94,
                box_nms_thresh=0.4,
                crop_n_layers=1,
                crop_n_points_downscale_factor=2,
            )

            outputs = []

            for image in images:
                # Step 1: Move to CPU and convert to NumPy
                image_np = image.cpu().numpy()

                # Step 2: Transpose to (H, W, C)
                image_np = np.transpose(image_np, (1, 2, 0))

                # Step 3: Convert to uint8 if necessary (SAM expects 0–255 uint8 images)
                if image_np.dtype != np.uint8:
                    image_np = (image_np * 255).clip(0, 255).astype(np.uint8)

                masks = mask_generator.generate(image_np)
                sorted_masks = sorted(
                    masks, key=lambda m: m["stability_score"], reverse=True
                )

                output = {
                    "boxes": torch.tensor(
                        np.array([bbox_to_xyxy(mask["bbox"]) for mask in sorted_masks])
                    ),
                    "scores": torch.tensor(
                        np.array([mask["stability_score"] for mask in sorted_masks])
                    ),
                    "masks": torch.tensor(
                        np.array([mask["segmentation"] for mask in sorted_masks])
                    ),
                }

                outputs.append(output)

    else:
        raise ValueError("Model is not a MaskRCNN or SAM model.")

    result_boxes = []
    result_scores = []

    for i, o in enumerate(outputs):
        # boxes = o["boxes"].cpu().numpy()
        # scores = o["scores"].cpu().numpy()  # confidence scores
        boxes = o["boxes"]
        scores = o["scores"]

        # get color width and height but ignore color channel
        w, h = images[i].size()[1:]
        # masks = outputs["masks"].cpu().numpy().squeeze(1)

        selected_proposals = filter_boxes(
            boxes,
            scores,
            w * h,
            iou_threshold=iou_threshold,
            min_size=min_size,
            max_size=max_size,
            minimum_score=minimum_score,
        )

        ids = [p.item() for p in selected_proposals]
        result_boxes.append(boxes[ids])
        result_scores.append(scores[ids])

    return result_boxes, result_scores


def filter_boxes(
    boxes,
    scores,
    image_size,
    iou_threshold=0.5,
    min_size=0.1,
    max_size=0.8,
    minimum_score=0.1,
):
    """
    Filters the generated bounding boxes:
    1. Removes bounding boxes cover too much or too little of the image
    2. Remove bounding boxes with a certainty score lower than the minimum_score
    3. For the remaining bounding boxes: When sorted by score, filter all bounding boxes with a lower score but a high IoU with a utilized bounding box.
    """
    selected_indices = []
    # remaining_indices = np.argsort(scores)[::-1]
    remaining_indices = torch.argsort(scores, descending=True)

    # iterate over the scores (they are in deceding order)
    while len(remaining_indices) > 0:
        current_idx = remaining_indices[0]
        box_i = boxes[remaining_indices[0]]
        # check whether the current box is too large or too small
        box_area = (box_i[2] - box_i[0]) * (box_i[3] - box_i[1])
        if box_area / image_size < min_size or box_area / image_size > max_size:
            remaining_indices = remaining_indices[1:]
            continue
        # check whether the current box has a score lower than the minimum score
        if scores[remaining_indices[0]] < minimum_score:
            remaining_indices = remaining_indices[1:]
            continue

        selected_indices.append(remaining_indices[0])
        # Keep only boxes with IoU <= threshold
        keep_indices = []
        for idx in remaining_indices[1:]:
            iou = calculate_iou(boxes[current_idx], boxes[idx])
            if iou <= iou_threshold:
                keep_indices.append(idx)

        remaining_indices = torch.tensor(keep_indices, device=remaining_indices.device)

    return selected_indices


def calculate_iou(bbox1, bbox2):
    x1_inter = max(bbox1[0], bbox2[0])
    y1_inter = max(bbox1[1], bbox2[1])
    x2_inter = min(bbox1[2], bbox2[2])
    y2_inter = min(bbox1[3], bbox2[3])

    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

    box1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    box2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0


def obj_enc_aggregation(aggregation, obj_encodings, img_encodings, only_objects=False):
    """
    Aggregates object encodings based on the specified method.

    Parameters:
        aggregation (str): The aggregation method ('sum', 'max', 'concat').
        obj_encodings (torch.Tensor): Object encodings.
        img_encodings (torch.Tensor): Image encodings.
        only_objects (bool): If True, only aggregate object encodings, otherwise combine with image encodings.

    Returns:
        torch.Tensor: Aggregated encodings.
    """
    if aggregation == "sum":
        if only_objects:
            return obj_encodings.sum(dim=1)
        else:
            return obj_encodings.sum(dim=1) + img_encodings
    elif aggregation == "max":
        if only_objects:
            return obj_encodings.max(dim=1)[0]
        else:
            return torch.max(obj_encodings.max(dim=1)[0], img_encodings)
    elif aggregation == "concat":
        if only_objects:
            raise ValueError("Concatenation is not supported for only_objects")
        else:
            return torch.cat(
                (img_encodings, obj_encodings.reshape(obj_encodings.size(0), -1)), dim=1
            )
    elif aggregation == "sum_count":
        if only_objects:
            summed = obj_encodings.sum(dim=1)
            counts = torch.count_nonzero(obj_encodings, dim=1)
            return torch.stack((summed, counts), dim=-1).view(summed.size(0), -1)
        else:
            summed = obj_encodings.sum(dim=1) + img_encodings
            counts = (
                torch.count_nonzero(obj_encodings, dim=1) + (img_encodings != 0).long()
            )
            res = torch.stack((summed, counts), dim=-1)
            return res.view(res.size(0), -1)
    elif aggregation == "count":
        if only_objects:
            return torch.count_nonzero(obj_encodings, dim=1).float()
        else:
            return (
                torch.count_nonzero(obj_encodings, dim=1) + (img_encodings != 0).long()
            ).float()
    else:
        raise ValueError(f"Unsupported aggregation method: {aggregation}")


def persistent_indices(full_dataset, data_dir, seed):
    """
    If a split_indices.pt file exists in data_dir, load the train and test indices from it.
    Otherwise, create a new random split (80% train, 20% test), save it to split_indices.pt, and return the datasets.
    Ensures that the train/test split is persistent across different runs.
    """
    if os.path.exists(os.path.join(data_dir, "split_indices.pt")):
        indices = torch.load(os.path.join(data_dir, "split_indices.pt"))
        train_indices = indices["train"]
        test_indices = indices["test"]
    else:
        indices = torch.randperm(
            len(full_dataset), generator=torch.Generator().manual_seed(seed)
        )
        train_size = int(0.8 * len(full_dataset))
        # Split the indices into train and test sets
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]
        torch.save(
            {"train": train_indices, "test": test_indices},
            os.path.join(data_dir, "split_indices.pt"),
        )

    train_dataset = Subset(full_dataset, train_indices)
    test_dataset = Subset(full_dataset, test_indices)

    return train_dataset, test_dataset


def seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # For deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
