import os
import torch 
from PIL import Image 
import torchvision 
from torchvision import transforms
from torch.utils.data import DataLoader
from matting_dataset import MattingDataset, Rescale, ToTensor, Normalize, ToTrainArray, ConvertImageDtype
from src.trainer import supervised_training_iter, soc_adaptation_iter
from src.models.modnet import MODNet 

def load_images(image_paths):
    transform = transforms.Compose([
        transforms.Resize((512, 512)), 
        transforms.ToTensor(),
    ])
    images = [transform(Image.open(path)) for path in image_paths]
    return torch.stack(images)

def train_model(modnet, dataloader, test_images, total_epochs, learning_rate):
    optimizer = torch.optim.SGD(modnet.parameters(), lr=learning_rate, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(0.25 * total_epochs), gamma=0.1)

    for epoch in range(total_epochs):
        for idx, (image, trimap, gt_matte) in enumerate(dataloader):
            semantic_loss, detail_loss, matte_loss = supervised_training_iter(modnet, optimizer, image.cuda(), trimap.cuda(), gt_matte.cuda()) 
            
        lr_scheduler.step()

        with torch.no_grad():
            _, _, debugImages = modnet(test_images.cuda(), True)
            for idx, img in enumerate(debugImages):
                saveName = "eval_%g_%g.jpg"%(idx, epoch+1)
                torchvision.utils.save_image(img, os.path.join(evalPath, saveName))

        print("Epoch done: " + str(epoch)) 

if __name__ == '__main__':
    transform = transforms.Compose([
        Rescale(512),
        ToTensor(),
        ConvertImageDtype(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ToTrainArray()
    ])
        
    batch_size = 16
    total_epochs = 40  
    learning_rate = 0.01
    
    mattingDataset = MattingDataset(transform=transform) 
    dataloader = DataLoader(mattingDataset, batch_size=batch_size, shuffle=True)

    modnet = torch.nn.DataParallel(MODNet()).cuda()  

    evalPath = 'dataset/UGD-12k/result'
    if not os.path.isdir(evalPath):
        os.makedirs(evalPath)

    test_image_paths = ['dataset/PPM-100/train/fg/14429083354_23c8fddff5_o.jpg']
    test_images = load_images(test_image_paths)

    train_model(modnet, dataloader, test_images, total_epochs, learning_rate) 
    
    torch.save(modnet.state_dict(), 'pretrained/UGD-12k_trained_model.pth')

