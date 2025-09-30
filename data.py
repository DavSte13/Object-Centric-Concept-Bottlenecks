from torch.utils.data import Dataset, random_split
import numpy as np
import os
import json
import torch
from PIL import Image
import torchvision.transforms as T
from torchvision.datasets import SUN397, VOCDetection, CIFAR100
from collections import defaultdict, Counter


class CocoDataset(Dataset):
    def __init__(
        self, data_dir, train_val, transform=None, use_crowd_annotations=False
    ):
        self.image_dir = os.path.join(data_dir, "images", f"{train_val}2017")
        self.annotation_dir = os.path.join(data_dir, "annotations")

        if transform is None:
            self.transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])
        else:
            self.transform = transform
        self.use_crowd_annotations = use_crowd_annotations

        self.image_files = os.listdir(self.image_dir)
        self.annotation_file = os.path.join(
            self.annotation_dir, f"instances_{train_val}2017.json"
        )
        self.category_info = (
            None  # info about supercategories and categories in the image
        )

        self.extract_annotations()

    def __len__(self):
        return len(self.image_files)

    def extract_annotations(self):
        with open(self.annotation_file, "r") as f:
            data_info = json.load(f)

        # process supercategorie information:
        supercategories = {}
        for category in data_info["categories"]:
            if category["supercategory"] not in supercategories.keys():
                supercategories[category["supercategory"]] = []
            supercategories[category["supercategory"]].append(category["id"])

        # replace the string by an id:
        super_labels = {}
        for i, key in enumerate(supercategories.keys()):
            super_labels[i] = supercategories[key]

        # create a dict with category ids as keys and supercategory ids as values
        category_to_super = {}
        for key in super_labels.keys():
            for value in super_labels[key]:
                category_to_super[value] = key

        category_info = {}
        for annotation in data_info["annotations"]:
            # ignore crowd anntoations if specified
            if not self.use_crowd_annotations and annotation["iscrowd"]:
                continue
            # sort information per image
            if annotation["image_id"] not in category_info.keys():
                category_info[annotation["image_id"]] = np.zeros(12), np.zeros(90)
            # image info contains the categories and supercategories of objects in the image (as binary label, not as counts)
            category_info[annotation["image_id"]][0][
                category_to_super[annotation["category_id"]] - 1
            ] = 1
            category_info[annotation["image_id"]][1][annotation["category_id"] - 1] = 1

        self.category_info = category_info

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.image_files[index])
        # filenames are of the form 000000000001.jpg, get the image id from that
        img_id = int(self.image_files[index].split(".")[0])

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        # if there are no annotations for the image, return a zero vector as label
        if img_id not in self.category_info.keys():
            return image, np.zeros(12), np.zeros(90), img_id
        else:
            return (
                image,
                self.category_info[img_id][0],
                self.category_info[img_id][1],
                img_id,
            )


class SUNDataset(SUN397):
    def __init__(self, root, transform=None, target_transform=None, download=False):
        super().__init__(
            root=root,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
        if transform is None:
            self.transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])
        else:
            self.transform = transform

        self.filename_to_id = {}
        for i, filename in enumerate(self._image_files):
            self.filename_to_id[filename] = i

    def __getitem__(self, index):
        # as SUN does not have supercategories, categories and supercategories are the same
        img, label = super().__getitem__(index)
        # using the filename does not work, as it is a string and not an id.
        # img_id = str(self._image_files[index]).split("/")[-1].split(".")[0]

        label_exp = torch.zeros((397))
        label_exp[label] = 1
        return img, label_exp, label_exp, self.filename_to_id[self._image_files[index]]


class CIFAR100Dataset(CIFAR100):
    def __init__(self, root, transform=None, train=True, download=False):
        super().__init__(root=root, train=train, transform=transform, download=download)
        if self.transform is None:
            self.transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])

    def __getitem__(self, index):
        img, label = super().__getitem__(index)

        # img = self.transform(img) the parent dataloader applies the transform
        one_hot_label = torch.zeros(100)
        one_hot_label[label] = 1

        return img, one_hot_label, one_hot_label, index


class PascalVOCDataset(VOCDetection):
    def __init__(
        self,
        root,
        transform=None,
        target_transform=None,
        download=False,
        image_set="train",
        year="2012",
    ):
        super().__init__(root=root, download=download, year=year, image_set=image_set)
        if transform is None:
            self.transform = T.Compose([T.Resize((224, 224)), T.ToTensor()])
        else:
            self.transform = transform

        self.VOC_CLASS_TO_INDEX = {
            "aeroplane": 1,
            "bicycle": 2,
            "bird": 3,
            "boat": 4,
            "bottle": 5,
            "bus": 6,
            "car": 7,
            "cat": 8,
            "chair": 9,
            "cow": 10,
            "diningtable": 11,
            "dog": 12,
            "horse": 13,
            "motorbike": 14,
            "person": 15,
            "pottedplant": 16,
            "sheep": 17,
            "sofa": 18,
            "train": 19,
            "tvmonitor": 20,
        }

        self.filename_to_id = {}
        for i, filename in enumerate(self.images):
            self.filename_to_id[filename] = i

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        # as VOC does not have supercategories, categories and supercategories are the same
        img = self.transform(img)

        # parse all object categories and convert to indices
        l = [
            o["name"]
            for o in target["annotation"]["object"]
            if o["name"] != "background"
        ]
        l = list(set(l))
        l = [self.VOC_CLASS_TO_INDEX[i] for i in l]

        label = torch.zeros((20))
        for i in l:
            # correct for the fact that the labels are 1-indexed
            label[i - 1] = 1
        return img, label, label, index


def load_category_mapping(annotation_file):
    with open(annotation_file, "r") as f:
        coco_data = json.load(f)
    categories = coco_data["categories"]
    return {cat["id"]: cat["name"] for cat in categories}


def split_dataset(dataset, train_ratio=0.8):
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    return random_split(dataset, [train_size, test_size])


class COCOLogicDataset(Dataset):
    def __init__(
        self,
        annotation_file,
        image_dir,
        category_id_to_name,
        transform=None,
        filter_no_labels=True,
        exclusive_label=True,
        exclusive_match_only=True,
        log_statistics=False,
        version=10,
    ):
        """
        annotation_file: path to COCO annotations JSON
        image_dir: path to the images folder
        category_id_to_name: dict mapping COCO category id to names
        filter_no_labels: if True, drops images that satisfy no logical classes
        exclusive_label: if True, only assign first matching class as 1 (others 0)
        exclusive_match_only: if True, only include images that match exactly one logical class

            exclusive_match_only | exclusive_label | Resulting Effect
            False | False | Multi-label dataset, overlapping classes allowed.
            False | True | Overlapping images included, but only first class is used in label.
            True | False | Only images with one class are included, label is still multi-hot (only one 1).
            True | True | Clean single-class dataset, label is one-hot — ideal for multi-class classification.
        """

        self.image_dir = image_dir
        self.transform = transform
        self.exclusive_label = exclusive_label
        self.exclusive_match_only = exclusive_match_only
        self.version = version

        # Load COCO annotations
        with open(annotation_file, "r") as f:
            coco_data = json.load(f)

        self.imgs = {img["id"]: img for img in coco_data["images"]}
        self.annotations = coco_data["annotations"]

        self.image_to_categories = {}
        category_frequency = Counter()
        for ann in self.annotations:
            img_id = ann["image_id"]
            cat_id = ann["category_id"]
            cat_name = category_id_to_name[cat_id]
            self.image_to_categories.setdefault(img_id, set()).add(cat_name)
            category_frequency[cat_name] += 1

        # # logical class definitions
        if self.version == 10:
            self.logical_classes = [
                # 1. Ambiguous Pairs (Pet vs Ride Paradox). The image includes either a cat or a dog (but not both),
                # and either a bicycle or a motorcycle (but not both).
                (
                    "Ambiguous Pairs (Pet vs Ride Paradox)",
                    lambda cats: (("cat" in cats) ^ ("dog" in cats))
                    and (("bicycle" in cats) ^ ("motorcycle" in cats)),
                ),
                # 2. Pair of Pets. Exactly two of the following animal categories are present: a cat, a dog, or a bird.
                (
                    "Pair of Pets",
                    lambda cats: sum(c in cats for c in ["cat", "dog", "bird"]) == 2,
                ),
                # 3. Rural Animal Scene. The image includes one or more rural animals (cow, horse, or sheep) and no people.
                (
                    "Rural Animal Scene",
                    lambda cats: any(c in cats for c in ["cow", "horse", "sheep"])
                    and "person" not in cats,
                ),
                # 4. Conflicted Companions (Leash vs Licence). An image features either a dog or a car, but not both.
                (
                    "Conflicted Companions (Leash vs Licence)",
                    lambda cats: ("dog" in cats) ^ ("car" in cats),
                ),
                # 5. Animal Meet Traffic. The image contains a rural animal (horse, cow, or sheep) and a
                # traffic-related object (car, bus, or traffic light).
                (
                    "Animal Meets Traffic",
                    lambda cats: any(c in cats for c in ["horse", "cow", "sheep"])
                    and any(c in cats for c in ["car", "bus", "traffic light"]),
                ),
                # 6. Occupied Interior. The image includes furniture (a couch or chair) and at least one person.
                (
                    "Occupied Interior",
                    lambda cats: any(c in cats for c in ["couch", "chair"])
                    and "person" in cats
                    and sum(c == "person" for c in cats) == 1,
                ),
                # 7. Empty Seat. The image includes indoor furniture (a couch or chair) but no person is present.
                (
                    "Empty Seat",
                    lambda cats: any(c in cats for c in ["couch", "chair"])
                    and "person" not in cats,
                ),
                # 8. Odd Ride Out. Exactly one of the following categories is present: a bicycle, motorcycle, car, or bus.
                (
                    "Odd Ride Out",
                    lambda cats: sum(
                        c in cats for c in ["bicycle", "motorcycle", "bus", "car"]
                    )
                    == 1,
                ),
                # 9. Personal Transport XOR Car. A person is present alongside either a bicycle or a car — but not both.
                (
                    "Personal Transport XOR Car",
                    lambda cats: "person" in cats
                    and (("bicycle" in cats) ^ ("car" in cats)),
                ),
                # 10. Unlikely Breakfast Guests. The image shows a bowl (suggesting food) and at least one animal (dog, cat, horse, cow, or sheep).
                (
                    "Unlikely Breakfast Guests",
                    lambda cats: "bowl" in cats
                    and any(c in cats for c in ["dog", "cat", "horse", "cow", "sheep"]),
                ),
            ]
        else:
            raise Exception("COCOLogic other than 10 is not implemented")

        total_images = len(self.imgs)
        kept_images = 0
        class_counts = defaultdict(int)
        class_cooccurrence = Counter()

        self.image_ids = []
        for img_id in self.imgs:
            cats = self.image_to_categories.get(img_id, set())
            labels = [int(fn(cats)) for _, fn in self.logical_classes]

            if filter_no_labels and not any(labels):
                continue

            if exclusive_match_only and sum(labels) != 1:
                continue

            self.image_ids.append(img_id)
            kept_images += 1

            label_tuple = tuple(labels)
            class_cooccurrence[label_tuple] += 1

            for i, val in enumerate(labels):
                if val:
                    class_counts[self.logical_classes[i][0]] += 1

        if log_statistics:
            print(f"\nLogicalCOCODataset: Loaded {total_images} images.")
            print(
                f"Filtered to {kept_images} images after applying logical class filters.\n"
            )

            print("Per-Class Image Count:")
            for class_name, _ in self.logical_classes:
                print(f" - {class_name:<25}: {class_counts[class_name]}")

            print("\nTop 20 Class Co-occurrence Patterns:")
            for pattern, count in class_cooccurrence.most_common(20):
                pattern_str = ", ".join(
                    [self.logical_classes[i][0] for i, v in enumerate(pattern) if v]
                )
                print(f" - [{pattern_str or 'None'}] : {count} images")

            print("\nTop 20 Category Frequencies:")
            for cat, count in category_frequency.most_common(20):
                print(f" - {cat:<20}: {count} annotations")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.imgs[img_id]
        img_path = os.path.join(self.image_dir, img_info["file_name"])

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        categories = self.image_to_categories.get(img_id, set())
        labels = [int(fn(categories)) for _, fn in self.logical_classes]

        if self.exclusive_label:
            exclusive = [0] * len(labels)
            for i, val in enumerate(labels):
                if val:
                    exclusive[i] = 1
                    break
            labels = exclusive

        label_index = torch.tensor(labels.index(1), dtype=torch.long)
        # return labels as one.hot and twice for consistency
        return image, torch.tensor(labels), torch.tensor(labels), img_id
