import torch
import os
import time


def save_model(model, optimizer, root_folder, file_name, hyperparameter_dict, best_model, last_epoch, verbose=False):
    
    os.makedirs(root_folder, exist_ok=True)
    model_full_path = os.path.join(root_folder, file_name+'.pth.tar')
    
    torch.save({
        'hyperparameters': hyperparameter_dict,
        'best_model': best_model,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'saved_time_unix': time.time(),
        'saved_time_asctime': time.asctime(),
        'last_epoch': last_epoch,
    }, model_full_path)
    
    if verbose:
        print(f'Checkpoint: {file_name} is saved successfully')
    
    
def load_model(model, optimizer, root_folder, file_name):
    
    model_full_path = os.path.join(root_folder, file_name+'.pth.tar')
    checkpoint = torch.load(model_full_path)
    
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    
    print(f'Checkpoint: {file_name} is loaded successfully')
    
    return checkpoint
