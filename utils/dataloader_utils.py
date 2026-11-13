import torch
from torch.utils.data import DataLoader

import albumentations as A

from dataset.dataset import seg_datasets

def get_loader(
    train_dir=None,
    test_dir=None,
    batch_size=8,
    train_transform=None,
    test_transform=None,
    num_workers=4,
    pin_memory=False
):
    
    """
    if train_dir is None:
        train_dir = './images/train'
        
    if test_dir is None:
        test_dir = './images/test'
        
    img_transforms = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        #A.RandomBrightnessContrast(p=0.5),
        A.Rotate(p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        A.Resize(512,512),
    ])

    test_transforms = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize(512, antialias=True)
    ])


    if train_transform is None:
        train_transform = img_transforms
        
    if test_transform is None:
        test_transform = test_transforms
        
        
    """
    
    train_dataset = seg_datasets(train_dir, train_transform)
    test_dataset = seg_datasets(test_dir, test_transform)
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=pin_memory,
        #collate_fn=lambda batch: tuple(zip(*batch)), 
    )

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=pin_memory,
        #collate_fn=lambda batch: tuple(zip(*batch)),
    )
    
    return train_dataloader, test_dataloader