import csv
import os
import zipfile
from io import BytesIO

import numpy as np
import requests
from PIL import Image
from torch.utils.data import Dataset
from tqdm import notebook


class TinyImageNet(Dataset):
    """
    Tiny ImageNet Dataset class.
    """

    def __init__(self, root, train=True, transform=None, download=False, train_split=0.7):
        self.root = root
        self.train = train
        self.transform = transform
        self.data_dir = "tiny-imagenet-200"

        os.makedirs(self.root, exist_ok=True)

        if download:
            self.download_and_extract_archive()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        self.image_paths = []
        self.targets = []

        idx_to_class, class_id = self.get_classes()

        self.classes = list(idx_to_class.values())

        train_path = os.path.join(self.root, self.data_dir, "train")
        for class_dir in os.listdir(train_path):
            train_images_path = os.path.join(train_path, class_dir, "images")
            for image in os.listdir(train_images_path):
                if image.endswith(".JPEG"):
                    self.image_paths.append(os.path.join(train_images_path, image))
                    self.targets.append(class_id[class_dir][0])
        
        # val images
        val_path = os.path.join(self.root, self.data_dir, "val")
        val_images_path = os.path.join(val_path, "images")
        with open(os.path.join(val_path, "val_annotations.txt")) as val_ann:
            for line in csv.reader(val_ann, delimiter="\t"):
                self.image_paths.append(os.path.join(val_images_path, line[0]))
                self.targets.append(class_id[line[1]][0])
        
        self.indices = np.arange(len(self.targets))

        random_seed = 42
        np.random.seed(random_seed)
        np.random.shuffle(self.indices)

        split_idx = int(len(self.indices) * train_split)
        self.indices = self.indices[:split_idx] if train else self.indices[split_idx:]

        if self.train:        
            self.data = []
            for i in self.image_paths[0:split_idx]:    
                img = plt.imread(i)
                if img.shape == (64,64,3):    
                    self.data.append(img)
                else:
                    im = plt.imread(i)
                    im2 = cv2.merge((im,im,im))
                    self.data.append(im2)
            
            self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
            self.data = self.data.transpose((0, 2, 3, 1))  # convert to H

        else:
            self.data = []
            for i in self.image_paths[split_idx:]:    
                img = plt.imread(i)
                if img.shape == (64,64,3):    
                    self.data.append(img)
                else:
                    im = plt.imread(i)
                    im2 = cv2.merge((im,im,im))
                    self.data.append(im2)
            
            self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
            self.data = self.data.transpose((0, 2, 3, 1))  # convert to H



    def get_classes(self):
        """
        Get class labels mapping
        """
        id_dict = {}
        all_classes = {}
        for i, line in enumerate(open(os.path.join(self.root, "tiny-imagenet-200/wnids.txt"), "r")):
            id_dict[line.replace("\n", "")] = i

        idx_to_class = {}
        class_id = {}
        for i, line in enumerate(open(os.path.join(self.root, "tiny-imagenet-200/words.txt"), "r")):
            n_id, word = line.split("\t")[:2]
            all_classes[n_id] = word
        for key, value in id_dict.items():
            idx_to_class[value] = all_classes[key].replace("\n", "").split(",")[0]
            class_id[key] = (value, all_classes[key])

        return idx_to_class, class_id

    def _check_integrity(self) -> bool:
        """
        Check if Tiny ImageNet data already exists.
        """
        return os.path.exists(os.path.join(self.root, self.data_dir))

    def download_and_extract_archive(self):
        """
        Download and extract Tiny ImageNet data.
        """
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        res = requests.get("http://cs231n.stanford.edu/tiny-imagenet-200.zip", stream=True)
        print("Downloading Tiny ImageNet Data")

        with zipfile.ZipFile(BytesIO(res.content)) as zip_ref:
            for file in notebook.tqdm(iterable=zip_ref.namelist(), total=len(zip_ref.namelist())):
                zip_ref.extract(member=file, path=self.root)

    def __getitem__(self, idx):
        image_idx = self.indices[idx]
        filepath = self.image_paths[image_idx]
        img = Image.open(filepath)
        img = img.convert("RGB")
        target = self.targets[image_idx]

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.indices)