import splice
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image
import torch.nn.functional as F
import torchvision.transforms.v2 as T
from torch.utils.data import Subset
import numpy as np
import argparse
import os
from PIL import Image
from tqdm import tqdm
import h5py

from data import CocoDataset, SUNDataset, PascalVOCDataset, CorelDataset
from torchvision.models.detection import maskrcnn_resnet50_fpn
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from utils import generate_object_propsals, gather_object_embeddings, CLASS_NUMBERS, peristent_indices
from arg_utils import add_general_args, add_generation_args, add_object_centric_args, add_dataset_args

from rtpt import RTPT


def generate_concepts(dataset, model, preprocessor, data_dir, encoding_dir, batch_size, device, initials, seed, l1_penalty):

    os.makedirs(encoding_dir, exist_ok=True)
    os.makedirs(os.path.join(encoding_dir, dataset), exist_ok=True)

    if dataset == "coco":
        train_dataset = CocoDataset(data_dir, transform=preprocessor, train_val="train")
        test_dataset = CocoDataset(data_dir, transform=preprocessor, train_val="val")
    elif dataset == "sun":
        full_dataset = SUNDataset(root=data_dir, transform=preprocessor, download=True)
        train_dataset, test_dataset = peristent_indices(full_dataset, data_dir, seed=seed)

    elif dataset == "voc":
        train_dataset = PascalVOCDataset(data_dir, year='2012', transform=None, image_set='train', download=True)
        test_dataset = PascalVOCDataset(data_dir, year='2012', transform=None, image_set='val', download=True)
    else:
        raise ValueError(f"Dataset {dataset} not implemented yet.")
    
    # Setup dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, 
                             num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=4, pin_memory=True)
    
    # Calculate dataset sizes
    train_size = len(train_dataset)
    test_size = len(test_dataset)
    embedding_dim = 10000  # SpLiCE embedding dimension
    clip_embedding_dim = 512  # SpLiCE embedding dimension

    supercategory_dim, category_dim = CLASS_NUMBERS[dataset]  # Number of supercategories and categories in the dataset


    # Create HDF5 file and pre-allocate datasets
    with h5py.File(os.path.join(encoding_dir, dataset, f"splice_embeddings_{l1_penalty}.h5"), 'w') as f:
        # Create datasets with chunks for efficient writing
        f.create_dataset('train_embeddings', shape=(train_size, embedding_dim),
                        dtype=np.float32, chunks=(batch_size, embedding_dim))
        f.create_dataset('test_embeddings', shape=(test_size, embedding_dim),
                        dtype=np.float32, chunks=(batch_size, embedding_dim))

        # Create datasets with chunks for efficient writing for clip encs
        f.create_dataset('train_clip_embeddings', shape=(train_size, clip_embedding_dim),
                        dtype=np.float32, chunks=(batch_size, clip_embedding_dim))
        f.create_dataset('test_clip_embeddings', shape=(test_size, clip_embedding_dim),
                        dtype=np.float32, chunks=(batch_size, clip_embedding_dim))

        # Create datasets for structured coco categories
        f.create_dataset('train_category', shape=(train_size, category_dim),
                        dtype=np.int64, chunks=(batch_size, category_dim))
        f.create_dataset('test_category', shape=(test_size, category_dim),
                        dtype=np.int64, chunks=(batch_size, category_dim))
        f.create_dataset('train_supercategory', shape=(train_size, supercategory_dim),
                        dtype=np.int64, chunks=(batch_size, supercategory_dim))
        f.create_dataset('test_supercategory', shape=(test_size, supercategory_dim),
                        dtype=np.int64, chunks=(batch_size, supercategory_dim))
        
        # Create datasets for image ids
        f.create_dataset('train_img_ids', shape=(train_size, 1),
                        dtype=np.int64, chunks=(batch_size, 1))
        f.create_dataset('test_img_ids', shape=(test_size, 1),
                        dtype=np.int64, chunks=(batch_size, 1))
        

        rtpt = RTPT(name_initials=initials, experiment_name="Embedding generation", max_iterations=len(train_loader)+len(test_loader))
        rtpt.start()

        # Process training set
        print(f"Computing train set embeddings...")
        start_idx = 0
        with torch.no_grad():
            for i, (images, super_categories, categories, img_ids) in enumerate(tqdm(train_loader)):
                rtpt.step()
                images = images.to(device)
                embeddings, clip_encs = model.encode_image(images)

                # Write batch directly to file
                end_idx = start_idx + len(images)
                f['train_embeddings'][start_idx:end_idx] = embeddings.cpu().numpy()
                f['train_clip_embeddings'][start_idx:end_idx] = clip_encs.cpu().numpy()
                f['train_category'][start_idx:end_idx] = categories.numpy()
                f['train_supercategory'][start_idx:end_idx] = super_categories.numpy()
                f['train_img_ids'][start_idx:end_idx] = np.expand_dims(img_ids.numpy(), axis=1)
                start_idx = end_idx

        # Process test set
        print(f"Computing test set embeddings...")
        start_idx = 0
        with torch.no_grad():
            for i, (images, super_categories, categories, img_ids) in enumerate(tqdm(test_loader)):
                rtpt.step()
                images = images.to(device)
                embeddings, clip_encs = model.encode_image(images)
                
                # Write batch directly to file
                end_idx = start_idx + len(images)
                f['test_embeddings'][start_idx:end_idx] = embeddings.cpu().numpy()
                f['test_clip_embeddings'][start_idx:end_idx] = clip_encs.cpu().numpy()
                f['test_category'][start_idx:end_idx] = categories.numpy()
                f['test_supercategory'][start_idx:end_idx] = super_categories.numpy()
                f['test_img_ids'][start_idx:end_idx] = np.expand_dims(img_ids.numpy(), axis=1)
                start_idx = end_idx

    print(f"Done! Embeddings saved to {encoding_dir}/{dataset}/splice_embeddings_{l1_penalty}.h5")


def generate_obj_encodings(dataset, splice_model, obj_model, preprocessor, data_dir, batch_size, device, num_objects, min_object_size, max_object_size, min_score, max_iou_threshold, file_path, initials, seed):
    # generate object_centric encodings
    
    if dataset == "coco":
        train_dataset = CocoDataset(data_dir, transform=None, train_val="train")
        test_dataset = CocoDataset(data_dir, transform=None, train_val="val")
    elif dataset == "sun":
        full_dataset = SUNDataset(root=data_dir, transform=None, download=True)
        train_dataset, test_dataset = peristent_indices(full_dataset, data_dir, seed=seed)        
    elif dataset == "voc":
        train_dataset = PascalVOCDataset(data_dir, year='2012', transform=None, image_set='train', download=True)
        test_dataset = PascalVOCDataset(data_dir, year='2012', transform=None, image_set='val', download=True)
    else:
        raise ValueError(f"Dataset {dataset} not implemented yet.")
    
    
    # Setup dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, 
                             num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=4, pin_memory=True)

    # Calculate dataset sizes
    train_size = len(train_dataset)
    test_size = len(test_dataset)
    embedding_dim = 10000  # SpLiCE embedding dimension

    # Create HDF5 file and pre-allocate datasets
    with h5py.File(file_path, 'w') as f:
        # Create datasets with chunks for efficient writing
        f.create_dataset('train_obj_embeddings', shape=(train_size, num_objects, embedding_dim),
                        dtype=np.float32, chunks=(batch_size, num_objects, embedding_dim))
        f.create_dataset('test_obj_embeddings', shape=(test_size, num_objects, embedding_dim),
                        dtype=np.float32, chunks=(batch_size, num_objects, embedding_dim))
        
        # Create a dataset to keep track of the number of detected objects
        f.create_dataset('train_num_obj', shape=(train_size, 1),
                        dtype=np.int64, chunks=(batch_size, 1))
        f.create_dataset('test_num_obj', shape=(test_size, 1),
                        dtype=np.int64, chunks=(batch_size, 1))

        # Create datasets for structured coco labels
        f.create_dataset('train_obj_img_ids', shape=(train_size, 1),
                        dtype=np.int64, chunks=(batch_size, 1))
        f.create_dataset('test_obj_img_ids', shape=(test_size, 1),
                        dtype=np.int64, chunks=(batch_size, 1))
        
        rtpt = RTPT(name_initials=initials, experiment_name="Embedding generation", max_iterations=len(train_loader)+len(test_loader))
        rtpt.start()

        # Process training set
        print(f"Computing train set embeddings...")
        total_detected_objects = 0
        total_objects_step_1 = 0
        start_idx = 0
        with torch.no_grad():
            for i, (images, super_categories, categories, img_ids) in enumerate(tqdm(train_loader)):
                rtpt.step()
                images = images.to(device)

                embeddings = []
                object_counts = []

                # Generate object proposals
                boxes, scores = generate_object_propsals(images, obj_model, min_object_size, max_object_size, max_iou_threshold)

                for j in range(len(images)):
                    obj_embeddings, num_objs = gather_object_embeddings(boxes[j], scores[j], min_score, num_objects, images[j], device, splice_model, embedding_dim, preprocessor)

                    embeddings.append(obj_embeddings)
                    object_counts.append(num_objs)
                    total_objects_step_1 += len(boxes[j])
                embeddings = torch.stack(embeddings)
                object_counts = torch.tensor(object_counts).unsqueeze(1)

                # Write batch directly to file
                end_idx = start_idx + len(images)
                f['train_obj_embeddings'][start_idx:end_idx] = embeddings.cpu().numpy()
                f['train_num_obj'][start_idx:end_idx] = object_counts.cpu().numpy()
                f['train_obj_img_ids'][start_idx:end_idx] = np.expand_dims(img_ids.numpy(), axis=1)
                start_idx = end_idx
                total_detected_objects += sum(object_counts.numpy())
                if i == 0:
                    print(f"Processed the first batch, with an average of {total_detected_objects/((i+1) * batch_size)} objects so far.")
                    print(f"Average number of objects detected in step 1: {total_objects_step_1/((i+1) * batch_size)}")

        print(f"Average number of detected objects per image in the train set: {total_detected_objects / len(train_loader.dataset)}")

        # Process test set
        print(f"Computing test set embeddings...")
        total_detected_objects = 0
        start_idx = 0
        with torch.no_grad():
            for i, (images, super_categories, categories, img_ids) in enumerate(tqdm(test_loader)):
                rtpt.step()
                images = images.to(device)

                embeddings = []
                object_counts = []

                # Generate object proposals
                boxes, scores = generate_object_propsals(images, obj_model, min_object_size, max_object_size, max_iou_threshold)
                for j in range(len(images)):
                    obj_embeddings, num_objs = gather_object_embeddings(boxes[j], scores[j], min_score, num_objects, images[j], device, splice_model, embedding_dim, preprocessor)

                    embeddings.append(obj_embeddings)
                    object_counts.append(num_objs)
                embeddings = torch.stack(embeddings)
                object_counts = torch.tensor(object_counts).unsqueeze(1)

                # Write batch directly to file
                end_idx = start_idx + len(images)
                f['test_obj_embeddings'][start_idx:end_idx] = embeddings.cpu().numpy()
                f['test_num_obj'][start_idx:end_idx] = object_counts.cpu().numpy()
                f['test_obj_img_ids'][start_idx:end_idx] = np.expand_dims(img_ids.numpy(), axis=1)
                start_idx = end_idx
                total_detected_objects += sum(object_counts.numpy())
        
        print(f"Average number of detected objects per image in the test set: {total_detected_objects / len(test_loader.dataset)}")

    print(f"Done! Embeddings saved to {file_path}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generate Splice concepts for a set of images')
    parser = add_general_args(parser)
    parser = add_dataset_args(parser)
    parser = add_generation_args(parser)
    parser = add_object_centric_args(parser)
    args = parser.parse_args()

    splicemodel = splice.load(args.model, args.vocabulary, args.size, l1_penalty=args.l1_penalty, device=args.device, return_weights=True)
    preprocessor = splice.get_preprocess(args.model)

    if args.use_object_concepts:
        if args.object_detection_model == "sam":
            # Load the SAM model
            model = sam_model_registry['vit_h'](checkpoint='sam_vit_h_4b8939.pth')
        elif args.object_detection_model == "mask-rcnn":
            # Load the MaskRCNN model
            model = maskrcnn_resnet50_fpn(pretrained=True)
        else:
            raise ValueError(f"Object detection model {args.object_detection_model} not implemented yet.")

        model.eval()
        model.to(args.device)

        # remove the ToTensor from the splice processor, as the object crops are already tensors
        custom_preprocessor = T.Compose([
            T.Resize(size=224, interpolation=T.InterpolationMode.BICUBIC, max_size=None, antialias=True),
            T.CenterCrop(size=(224, 224)),
            T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))])


        file_path = os.path.join(args.encoding_dir, args.dataset, args.object_encoding_dir, f'obj_{args.num_objects}_min_{args.min_object_size}_max_{args.max_object_size}_score_{args.min_score}_iou_{args.max_iou_threshold}_model_{args.object_detection_model}.h5')
        
        # create file folder if it does not exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Generate object-centric encodings
        generate_obj_encodings(args.dataset, splicemodel, model, custom_preprocessor, args.data_dir, args.batch_size, args.device, args.num_objects, args.min_object_size, args.max_object_size, args.min_score, args.max_iou_threshold, file_path, initials=args.initials, seed=args.seed)
    else:
        generate_concepts(args.dataset, splicemodel, preprocessor, args.data_dir, args.encoding_dir, args.batch_size, args.device, initials=args.initials, seed=args.seed, l1_penalty=args.l1_penalty)




