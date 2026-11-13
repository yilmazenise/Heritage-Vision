import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import random_split
    
import torchvision.transforms.functional as tF

from dataset.dataset import seg_datasets

from utils.dataloader_utils import get_loader
from utils.save_load_utils import save_model, load_model
from utils.evaluation_utils import calculate_miou, check_acc
from utils.visualization_utils import save_pred_masks, visualize_segmentation, plot_segmentation
                                                                     
import albumentations as A

import os
import time
import random
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageOps

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'USING DEVICE: {device}')

hyperparameters = {
    'test_dir' : './images/test',
    'train_dir' : './images/train',
    'seed' : 9999,
    'dataloader' :{
        'batch_size' : 8,
        'num_workers' : 4,
        'pin_memory' : False, 
    },
    'optimizer' :{
        'lr' : 1e-3,
        'betas' : (0.9, 0.999),
        'weight_decay' : 1e-4,
    },
    'epochs' : 150,
    'training_size' : 0,
    'test_size': 0,
    'gradient_clipping' : 1.0,
    'unet': {
        'in_channels': 3,
        'out_channels': 1,
        'features': 64,
        'height': 320,
        'width': 320,
        'pretrained_name': '',
    }, 
    'checkpoints_dir':'./checkpoints/pretrained/'
}

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(hyperparameters['seed'])


train_transforms = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Rotate(p=0.5),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    A.Resize(512, 512),
])

test_transforms = A.Compose([
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    A.Resize(512, 512),
])

train_dataloader, test_dataloader = get_loader(
    train_dir = hyperparameters['train_dir'],
    test_dir = hyperparameters['test_dir'],
    batch_size = hyperparameters['dataloader']['batch_size'],
    train_transform = train_transforms,
    test_transform = test_transforms,
    num_workers = hyperparameters['dataloader']['num_workers'],
    pin_memory = hyperparameters['dataloader']['pin_memory']
)

from backbones_unet.model.unet import Unet

model = Unet(
    backbone='convnext_base', # backbone network name
    in_channels=3,            # input channels (1 for gray-scale images, 3 for RGB, etc.)
    num_classes=1,            # output channels (number of classes in your dataset)
)

model.to(device)

criterion = nn.MSELoss(reduction='none')


color_mapping = {
    0: (0, 0, 0),       # Walls
    1: (255, 80, 80),     # Iwan
    2: (80, 80, 255),     # Room
    3: (255, 255, 255), # Background
}

def loss_fn(img_input, pred_student, mask, alpha=0.01):
    
    loss_by_pixel = criterion(pred_student, mask)
    loss = loss_by_pixel.mean()
    
    return loss

learning_rate = hyperparameters['optimizer']['lr']
betas = hyperparameters['optimizer']['betas']

optimizer = torch.optim.Adam(
    model.parameters(),
    lr = learning_rate,
    betas = betas, 
)

def train_fn(
    epoch,
    model,
    dataloader, 
    optimizer, 
    criterion, 
    device,
    gradient_clipping,
):
    
    model.train()
    
    compute_avg = lambda x: sum(x) / len(x)
    total_batches = len(dataloader)
    write_interval = total_batches // 4
    
    avg_loss = []
    epochs = hyperparameters['epochs']
    
    with tqdm(total=hyperparameters['training_size'], desc=f'Train/Epoch {epoch}/{epochs}', unit='img') as pbar:
        for idx, (img, mask) in enumerate(dataloader):
            img = img.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()

            pred_mask = model(img)

            loss = criterion(img_input=img, 
                              pred_student=pred_mask, 
                              mask=mask)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            optimizer.step()

            # Record Results
            avg_loss.append(loss.item())
            
            pbar.update(img.shape[0])
        
    return compute_avg(avg_loss) 


@torch.no_grad()
def eval_fn(
    epoch,
    model, 
    dataloader, 
    criterion, 
    device,
    color_mapping,
):
    model.eval()
    
    # Average Loss and mIoU
    avg_loss = []
    avg_mIoU = []
    
    #Accuracy
    total_correct = 0
    total_pixel = 0

    compute_avg = lambda x: sum(x) / len(x)
    total_batches = len(dataloader)
    write_interval = total_batches // 4
    
    epochs = hyperparameters['epochs']
    
    with tqdm(total=hyperparameters['test_size'], desc=f'Eval/Epoch {epoch}/{epochs}', unit='img') as pbar:
        for idx, (img, mask) in enumerate(dataloader):
            img = img.to(device)
            mask = mask.to(device)

            #Accuracy
            num_correct, num_pixel = check_acc(img, mask, model, device)
            total_correct += num_correct
            total_pixel += num_pixel

            pred_mask = model(img)
            loss = criterion(img_input=img, 
                              pred_student=pred_mask, 
                              mask=mask)

            # mIoU
            miou = calculate_miou(pred_mask, mask, color_mapping)

            avg_loss.append(loss.item())
            avg_mIoU.append(miou.item())
            
            pbar.update(img.shape[0])

    acc = 100*(num_correct/num_pixel)
    return compute_avg(avg_loss), compute_avg(avg_mIoU), acc

best_model = {
    'best_state_dict' : None,
    'best_epoch' : None,
    'best_eval_loss' : None,
    'best_eval_miou' : 0.0,
    'best_eval_acc' : None,
}

def model_train_start():
    model.to(device)
    
    for epoch in range(1, hyperparameters['epochs']+1):  
        
        train_loss = train_fn(
            epoch = epoch,
            model = model,
            dataloader = train_dataloader, 
            optimizer = optimizer, 
            criterion = loss_fn, 
            device = device,
            gradient_clipping = hyperparameters['gradient_clipping'],
        )
        
        eval_loss, eval_miou, eval_acc = eval_fn(
            epoch = epoch,
            model = model, 
            dataloader = test_dataloader, 
            criterion = loss_fn, 
            device = device,
            color_mapping = color_mapping,
        )

        print(f""""
        Epoch: {epoch}, 
        Average Train Loss: {train_loss:.4f} 
        Average Test Loss: {eval_loss:.4f} 
        Average Test mIoU: {eval_miou:.2f}
        Average Test Accuracy: {eval_acc} %""")
        
        if eval_miou > best_model['best_eval_miou']:
            best_model['best_state_dict'] = model.state_dict().copy()
            best_model['best_epoch'] = epoch
            best_model['best_eval_loss'] = eval_loss
            best_model['best_eval_miou'] = eval_miou
            best_model['best_eval_acc'] = eval_acc
            
            save_model(
                model=model, 
                optimizer=optimizer, 
                root_folder=hyperparameters['checkpoints_dir'], 
                file_name='pretrained_best', 
                hyperparameter_dict=hyperparameters, 
                best_model=best_model, 
                last_epoch=epoch, 
                verbose=True)

model_train_start()

print(
    f"""
    Best of Model:
        - Epoch Number: {best_model['best_epoch']}
        - Evaluation Loss: {best_model['best_eval_loss']}
        - Evaluation mIoU: {best_model['best_eval_miou']}
        - Evaluation Accuracy: {best_model['best_eval_acc']}
    """
)