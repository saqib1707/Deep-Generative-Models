import os
import torch
from torch import Tensor
from pathlib import Path
from typing import List, Optional, Sequence, Union, Any, Callable
from torchvision.datasets.folder import default_loader
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CelebA
import zipfile
import glob
import random
import cv2
import PIL


# Add your custom dataset class here
class ChestXRay(Dataset):
    def __init__(self, img_paths, transform=None):
        self.img_paths = img_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = cv2.imread(img_path)
#         print(img.shape)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # img = PIL.Image.open(img_path)
        img = PIL.Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, 0


class MyCelebA(CelebA):
    """
    A work-around to address issues with pytorch's celebA dataset class.
    
    Download and Extract
    URL : https://drive.google.com/file/d/1m8-EBPgi5MRubrm6iQjafK2QMHDBMSfJ/view?usp=sharing
    """
    
    def _check_integrity(self) -> bool:
        return True
    
    

class OxfordPets(Dataset):
    """
    URL = https://www.robots.ox.ac.uk/~vgg/data/pets/
    """
    def __init__(self, 
                 data_path: str, 
                 split: str,
                 transform: Callable,
                **kwargs):
        self.data_dir = Path(data_path) / "OxfordPets"        
        self.transforms = transform
        imgs = sorted([f for f in self.data_dir.iterdir() if f.suffix == '.jpg'])
        
        self.imgs = imgs[:int(len(imgs) * 0.75)] if split == "train" else imgs[int(len(imgs) * 0.75):]
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img = default_loader(self.imgs[idx])
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        return img, 0.0 # dummy datat to prevent breaking 

class VAEDataset(LightningDataModule):
    """
    PyTorch Lightning data module 

    Args:
        data_dir: root directory of your dataset.
        train_batch_size: the batch size to use during training.
        val_batch_size: the batch size to use during validation.
        patch_size: the size of the crop to take from the original images.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This can improve performance on GPUs.
    """

    def __init__(
        self,
        data_path: str,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        patch_size: Union[int, Sequence[int]] = (256, 256),
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.data_dir = data_path
        print("Data path:", self.data_dir)
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None) -> None:
#       =========================  OxfordPets Dataset  =========================
            
#         train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
#                                               transforms.CenterCrop(self.patch_size),
# #                                               transforms.Resize(self.patch_size),
#                                               transforms.ToTensor(),
#                                                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        
#         val_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
#                                             transforms.CenterCrop(self.patch_size),
# #                                             transforms.Resize(self.patch_size),
#                                             transforms.ToTensor(),
#                                               transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

#         self.train_dataset = OxfordPets(
#             self.data_dir,
#             split='train',
#             transform=train_transforms,
#         )
        
#         self.val_dataset = OxfordPets(
#             self.data_dir,
#             split='val',
#             transform=val_transforms,
#         )
        
#       =========================  CelebA Dataset  =========================
    
#         train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
#                                               transforms.CenterCrop(148),
#                                               transforms.Resize(self.patch_size),
#                                               transforms.ToTensor(),])
        
#         val_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
#                                             transforms.CenterCrop(148),
#                                             transforms.Resize(self.patch_size),
#                                             transforms.ToTensor(),])
        
        # self.train_dataset = MyCelebA(
        #     self.data_dir,
        #     split='train',
        #     transform=train_transforms,
        #     download=False,
        # )
        
        # # Replace CelebA with your dataset
        # self.val_dataset = MyCelebA(
        #     self.data_dir,
        #     split='test',
        #     transform=val_transforms,
        #     download=False,
        # )

#       =========================  ChestXRay Dataset  ========================= 

        # loading the training data
        train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.CenterCrop(1000),
                                                  transforms.Resize(self.patch_size), transforms.ToTensor(),])

        train_data_path = os.path.join(self.data_dir, "train/")
        print("Train Data Path:", train_data_path)
        train_img_paths = []
        
        for data_path in glob.glob(train_data_path + "*"):
            # print(data_path)
            train_img_paths.append(data_path)
        
        print("Number of train imgs:", len(train_img_paths))
        # train_img_paths = list(flatten(train_img_paths))
        # random.shuffle(train_img_paths)
        self.train_dataset = ChestXRay(train_img_paths, transform=train_transforms)
        
        # ------------------------------------------------------------------------------
        
        # loading the validation data
        val_transforms = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.CenterCrop(1000),
                                                transforms.Resize(self.patch_size), transforms.ToTensor(),])

        val_data_path = os.path.join(self.data_dir, "val/")
        print("Validation Data Path:", val_data_path)
        val_img_paths = []
        
        for data_path in glob.glob(val_data_path + "*"):
            val_img_paths.append(data_path)

        print("Number of validation imgs:", len(val_img_paths))
        self.val_dataset = ChestXRay(val_img_paths, transform=val_transforms)
        
        # ------------------------------------------------------------------------------
    
        # loading the test data
        test_transforms = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.CenterCrop(1000),
                                                transforms.Resize(self.patch_size), transforms.ToTensor(),])

        test_data_path = os.path.join(self.data_dir, "test/")
        print("Test Data Path:", test_data_path)
        test_img_paths = []
        
        for data_path in glob.glob(test_data_path + "*"):
            test_img_paths.append(data_path)

        print("Number of test imgs:", len(test_img_paths))
        self.test_dataset = ChestXRay(test_img_paths, transform=test_transforms)

#       ===============================================================
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )
    
    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        if self.test_dataset is None:
            return DataLoader(
                self.val_dataset,
                batch_size=self.val_batch_size,
                num_workers=self.num_workers,
                shuffle=True,
                pin_memory=self.pin_memory,
            )
        else:
            return DataLoader(
                self.test_dataset,
                batch_size=self.train_batch_size,
                num_workers=self.num_workers,
                shuffle=True,
                pin_memory=self.pin_memory,
            )
     