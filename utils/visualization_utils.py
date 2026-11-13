import torch
import os
import numpy as np
from PIL import Image
import random
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (6, 4)
plt.rcParams['axes.grid'] = False

def visualize_segmentation(predictions, color_mapping, from_tensor=False):
    
    if from_tensor:
        # Round the predictions to the nearest integer to get class labels
        predictions = torch.round(predictions).type(torch.LongTensor)
    
        predictions = predictions.numpy()[0] # Dummy channel dimension

    height, width = predictions.shape
    colored_mask = np.zeros((height, width, 3), dtype=np.uint8)

    for label, color in color_mapping.items():
        colored_mask[predictions == label] = color

    colored_mask = Image.fromarray(colored_mask)

    return colored_mask


@torch.no_grad()
def save_pred_masks(
    model,
    transform,
    save_folder,
    input_folder,
    color_mapping,
    hyperparameters
):
    
    #Create new folder for save pred masks if it doesn't exist
    os.makedirs(save_folder, exist_ok=True)
    model.eval()
    
    for filename in os.listdir(input_folder):
        if filename.endswith("Plan.jpg"):  # find image files  
            #Convert the image to PIL Image
            image = Image.open(os.path.join(input_folder, filename))
            
            #Convert the PIL Image to numpy array (dtype=float32)
            image = np.array(image).astype(np.float32)
            
            #Apple Albumentation transform function
            transformed_image = transform(image=image)["image"]
            
            #Convert the numpy array to torch tensor
            image = torch.from_numpy(transformed_image.copy().transpose((2,0,1))) 
            
            #Add a dummy batch dimension for model
            prediction = model(image.unsqueeze(0))
            
            #Round the float prediction number to closest int number
            prediction = torch.round(prediction).type(torch.LongTensor)
            
            #Remove the dummy batch dimension and color channel
            prediction = prediction.squeeze(0)
            prediction = prediction.numpy()[0] 
            
            #Create new array as same with prediction
            height, width = prediction.shape
            colored_prediction = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Assign colors to predicted segmentation by color mapping
            for label, color in color_mapping.items():
                colored_prediction[prediction == label] = color
                
            # Convert the numpy array image to PIL Image 
            colored_prediction = Image.fromarray(colored_prediction)
            
            #Change the file name
            mask_filename = filename.replace("Plan.jpg", "Pred_Mask.jpg")
            
            #Save the predicted mask
            colored_prediction.save(os.path.join(save_folder, mask_filename))


@torch.no_grad()
def plot_segmentation(model, dataset, color_mapping, save_fig, save_root_dir, save_filename, save_format, close_img=False):
    device = 'cpu'
    model.to(device)
    
    model.eval()
    os.makedirs(save_root_dir, exist_ok=True)

    # Number of outputs can vary for segmentation + classification
    # only segmentation: img, mask
    # with classification: img, mask, class_id
    sample = dataset[random.randint(0, len(dataset)-1)]
    img, mask = sample[0], sample[1]
    img=img.to(device)
    mask=mask.to(device)

    # TODO: handle varying number of outputs (segmentation, classification, etc.)
    #pred_mask = model(img.unsqueeze(0))[0]
    pred_mask = model(img.unsqueeze(0))
    pred_mask = pred_mask[0]

    actual_visualization = visualize_segmentation(mask, color_mapping, from_tensor=True)

    pred_visualization = visualize_segmentation(pred_mask, color_mapping, from_tensor=True)
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 12))

    fig.tight_layout()
    
    ax1.axis('off')
    ax1.set_title('Input Image')
    ax1.imshow(img.permute(1, 2, 0).numpy())
    
    ax2.axis('off')
    ax2.set_title('Actual Masks')
    ax2.imshow(actual_visualization)
    
    ax3.axis('off')
    ax3.set_title('Predicted Masks')
    ax3.imshow(pred_visualization)
    
    if save_fig and save_format == 'eps':
        _save_path = os.path.join(save_root_dir, f'{save_filename}.eps')
        plt.savefig(_save_path, format='eps')
    elif save_fig and save_format == 'png':
        _save_path = os.path.join(save_root_dir, f'{save_filename}.png')
        plt.savefig(_save_path, format='png', bbox_inches='tight')

    if close_img:
        plt.close()
        