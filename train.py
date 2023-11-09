import os
import glob
import torch 
import torchvision 
import numpy as np
from PIL import Image 
from eval import call_eval
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.trainer import supervised_training_iter, soc_adaptation_iter
from matting_dataset import MattingDataset, Rescale, ToTensor, Normalize, ToTrainArray, ConvertImageDtype

#Change it to the model type using:  
from src.models.modnet_old import MODNet #baseline MODNet
#from src.models.modnet import MODNet #ViTae 
#from src.models.modnet_tfi import MODNet #TFI

def get_latest_file(path):
    # Get a list of all files in the directory
    files = glob.glob(os.path.join(path, '*'))
    # Use the os.path.getmtime function to sort files by modification time
    files.sort(key=os.path.getmtime)
    # The last file in the list is the most recent one
    latest_file = files[-1]
    return latest_file

def load_images(image_paths):
    transform = transforms.Compose([
        transforms.Resize((512, 512)), 
        transforms.ToTensor(),
    ])
    images = [transform(Image.open(path)) for path in image_paths]
    return torch.stack(images)

def train_model(modnet, dataloader, total_epochs, learning_rate):
    optimizer = torch.optim.SGD(modnet.parameters(), lr=learning_rate, momentum=0.9)
    step_size = int(0.25 * total_epochs) if total_epochs >= 4 else 1
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)

    for epoch in range(total_epochs):
        for idx, (image, trimap, gt_matte) in enumerate(dataloader):
            semantic_loss, detail_loss, matte_loss = supervised_training_iter(modnet, optimizer, image.cuda(), trimap.cuda(), gt_matte.cuda()) 
            
        lr_scheduler.step()
        
        with torch.no_grad():
            call_eval(modnet)
            torch.save(modnet.state_dict(), f'pretrained/MODNet_epoch{epoch}.pth')
        #     _,_,debugImages = modnet(test_images.cuda(), True)
        #     for idx, img in enumerate(debugImages):
        #         saveName = "eval_%g_%g.jpg"%(idx,epoch+1)
        #         torchvision.utils.save_image(img, os.path.join(evalPath,saveName))

        print("Epoch done: " + str(epoch)) 

if __name__ == '__main__':
    transform = transforms.Compose([
        Rescale(512),
        ToTensor(),
        ConvertImageDtype(),
        Normalize((0.5), (0.5)),
        ToTrainArray()
    ])
        
    batch_size = 16
    total_epochs = [50]  
    learning_rate = 0.01
    
    mattingDataset = MattingDataset(transform=transform) 
    dataloader = DataLoader(mattingDataset, batch_size=batch_size, shuffle=True)
    print("dataset loaded")
    
    for epochs in total_epochs:
        modnet = torch.nn.DataParallel(MODNet()).cuda()
        train_model(modnet, dataloader, epochs, learning_rate) 

#     best_mse = float('inf')
#     best_mad = float('inf')
#     best_model = None
    
    # evalPath = 'dataset/UGD-12k/result3'
    # if not os.path.isdir(evalPath):
    #     os.makedirs(evalPath)
    
    # test_image_paths = ['dataset/UGD-12k/train/image/800036118.jpg']
    # test_images = load_images(test_image_paths) 
    # groundtruth_image_path = 'dataset/UGD-12k/eval/alpha/1000114288.png' 
    # groundtruth_image = Image.open(groundtruth_image_path)
    # # Resize the image
    # groundtruth_image = groundtruth_image.resize((512, 512))
    # groundtruth_image = np.array(groundtruth_image) 

#         # Calculate MSE and MAD
#         with torch.no_grad():
#             prediction_file_path = get_latest_file(evalPath)
#             prediction_image = np.array(Image.open(prediction_file_path).convert('L')) 
#             mse = mean_squared_error(groundtruth_image, prediction_image)
#             mad = mean_absolute_error(groundtruth_image, prediction_image) 

#         # Check if this model is better
#         if mse < best_mse and mad < best_mad:
#             best_mse = mse
#             best_mad = mad
#             best_model = modnet 
#             print(f'Best epochs: {epochs}, mse: {mse:.6f}, mad: {mad:.6f}')  

    # Save the best model 
    # torch.save(modnet.state_dict(), 'pretrained/tfi.pth')
 
