import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np


class seg_datasets(Dataset):
    def __init__(self, directory, transform = None):
        self.directory = directory
        self.transform = transform
        
        self.img_files = [img_file for img_file in os.listdir(directory) if (img_file.endswith('Plan.jpg'))]

        
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, index):
        
        selected_img_file = self.img_files[index]
        selected_mask_file = selected_img_file.replace("Plan.jpg", "Seg.jpg")
        
        
        #print(selected_img_file)
        #print(selected_mask_file)
        
        
        #convert the plan and mask from their path to PIL image. 
        plan = Image.open(os.path.join(self.directory, selected_img_file)) #RGB mode
        plan_ori = Image.open(os.path.join(self.directory, selected_img_file)).convert('1')
        mask = Image.open(os.path.join(self.directory, selected_mask_file)).convert('L')
        
        """
        print(f'\nimage: {plan}')
        plt.imshow(plan)
        plt.show()
        print(f'\nplan format: {plan.format}')
        print(f'\nplan size: {plan.size}')
        print(f'\nplan mode: {plan.mode}')
        
        print(f'\n\n\nmask: {mask}')
        print(f'\nmask format: {mask.format}')
        print(f'\nmask size: {mask.size}')
        print(f'\nmask mode: {mask.mode}')
        plt.imshow(mask)
        plt.show()
        """
        
        plan = np.array(plan).astype(np.float32)
        plan_ori = np.array(plan_ori).astype(np.float32)
        mask = np.array(mask).astype(np.float32)
 
        """
        print(f'\nplan: {plan}')
        print(f'\nplan.shape: {plan.shape}')
        print(f'\nplan.ndim: {plan.ndim}')
        print(f'\nplan.size: {plan.size}')
        
        print(f'\n\nmask: {mask}')
        print(f'\nmask.shape: {mask.shape}')
        print(f'\nmask.ndim: {mask.ndim}')
        print(f'\nmask.size: {mask.size}')
        """

        # Add dummy channel dimension
        mask = mask[..., np.newaxis]
        plan_ori = plan_ori[..., np.newaxis]
        
        mask_original = mask.copy()
        mask = np.zeros_like(mask).astype(np.float32)
         
        # SELECT MASKS ##
        ################
        # [  0.  29. 76. 150. 255.]
        mask[(mask_original <= 25.0)] = 0.0
        mask[(mask_original >= 26.0) & (mask_original <= 120.0)] = 1.0
        mask[(mask_original >= 121.0) & (mask_original <= 230.0)] = 2.0
        #mask[(mask_original >= 201.0) & (mask_original <= 230.0)] = 3.0
        mask[(mask_original >= 231.0)] = 3.0
        
        
        #plt.imshow(mask)
        #plt.show()
        
        
        # Apply image transformation (if any)
        if self.transform is not None:
            
            transformed = self.transform(image=plan, mask=mask)
            plan = transformed['image']
            mask = transformed['mask']
        
        # convert to tensor
        # (Width, Height, Channel) -> (Channel, Width, Height)
        plan = torch.from_numpy(plan.copy().transpose((2,0,1))) 
        mask = torch.from_numpy(mask.copy().transpose((2,0,1)))
        
        return plan, mask
        