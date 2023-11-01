import os
import cv2
import torch
import numpy as np
import pandas as pd
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from scipy import ndimage
from src.models.modnet import MODNet
from src import trainer as MODTrainer 

# Define dataset directory
dataset_dir = 'dataset/UGD-12k'

# Define data_csv file which contains image and matte file paths
data_csv = pd.read_csv('dataset/UGD-12k/dataset_paths.csv') 

# Define hyperparameters
batch_size = 16
learning_rate = 0.01
total_epochs = 40

# Create dataset class
class ModNetDataLoader(Dataset):
    def __init__(self, annotations_file, resize_dim, transform=None):
        self.img_labels =annotations_file
        self.transform=transform
        self.resize_dim=resize_dim
        self.current_idx = 0 

    def __len__(self):
        #return the total number of images
        return len(self.img_labels)

    def __getitem__(self, _):
        img_path = self.img_labels.iloc[self.current_idx,0]
        mask_path = self.img_labels.iloc[self.current_idx,1]
        print(self.current_idx)
        print("Image Path:", img_path)
        print("Matte Path:", mask_path)

        img = np.asarray(Image.open(img_path))

        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED) 
        mask = mask/255.0 

        if len(img.shape)==2:
            img = img[:,:,None]
        if img.shape[2]==1:
            img = np.repeat(img, 3, axis=2)
        elif img.shape[2]==4:
            img = img[:,:,0:3]

        if len(mask.shape)==3:
            mask = mask[:,:, 0]
        print(img.shape)
        print(mask.shape)
        #convert Image to pytorch tensor
        img = Image.fromarray(img)
        mask = Image.fromarray(mask)
        if self.transform:
            img = self.transform(img)
            trimap = self.get_trimap(mask)
            mask = self.transform(mask)

        img = self._resize(img)
        mask = self._resize(mask)
        trimap = self._resize(trimap, trimap=True)
        if img.size() != trimap.size() or img.size() != mask.size():
            raise ValueError("Image, trimap, and mask must have the same size")

        print(img.shape)
        print(mask.shape)
        print(trimap.shape)

        img = torch.squeeze(img, 0)
        mask = torch.squeeze(mask, 0)
        trimap = torch.squeeze(trimap, 1)
        self.current_idx += 1 

        return img, trimap, mask

    def get_trimap(self, alpha):
        # alpha \in [0, 1] should be taken into account
        # be careful when dealing with regions of alpha=0 and alpha=1
        fg = np.array(np.equal(alpha, 255).astype(np.float32))
        unknown = np.array(np.not_equal(alpha, 0).astype(np.float32)) # unknown = alpha > 0
        unknown = unknown - fg
        # image dilation implemented by Euclidean distance transform
        unknown = ndimage.distance_transform_edt(unknown == 0) <= np.random.randint(1, 20) 
        trimap = fg
        trimap[unknown] = 0.5
        return torch.unsqueeze(torch.from_numpy(trimap), dim=0)#.astype(np.uint8)

    def _resize(self, img, trimap=False):
        im = img[None, :, :, :]
        ref_size = self.resize_dim

        # resize image for input
        im_b, im_c, im_h, im_w = im.shape
        if max(im_h, im_w) < ref_size or min(im_h, im_w) > ref_size:
            if im_w >= im_h:
                im_rh = ref_size
                im_rw = int(im_w / im_h * ref_size)
            elif im_w < im_h:
                im_rw = ref_size
                im_rh = int(im_h / im_w * ref_size)
        else:
            im_rh = im_h
            im_rw = im_w

        im_rw = im_rw - im_rw % 32
        im_rh = im_rh - im_rh % 32
        if trimap == True:
            im = F.interpolate(im, size=(im_rh, im_rw), mode='area')
        else:
            im = F.interpolate(im, size=(im_rh, im_rw), mode='area')
        return im
    
# Define the transformation for data
transformer = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5))
])

# Create the dataset
data = ModNetDataLoader(data_csv, 512, transform=transformer)

# Create the data loader
dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)

# import model
modnet = torch.nn.DataParallel(MODNet()).cuda() 

# for evaluate progress
evalPath = 'dataset/PPM-100/val'
if not os.path.isdir(evalPath):
    os.makedirs(evalPath)
# pick 2 or more images here and store it for infer/eval later

# metaparams
optimizer = torch.optim.SGD(modnet.parameters(), lr=learning_rate, momentum=0.9)   # can try momentum=0.45 too
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(0.25 * total_epochs), gamma=0.1)


# Training starts here
for epoch in range(0, total_epochs):
    for idx, (image, trimap, gt_matte) in enumerate(dataloader):
        semantic_loss, detail_loss, matte_loss = MODTrainer.supervised_training_iter(modnet, optimizer, image.cuda(), trimap.cuda(), gt_matte.cuda())
    
    lr_scheduler.step()

    # eval for progress check and save images (here's where u visualize changes over training time)
    with torch.no_grad():
        _,_,debugImages = modnet(testImages.cuda(),True)
        for idx, img in enumerate(debugImages):
            saveName = "eval_%g_%g.jpg"%(idx,epoch+1)
            torchvision.utils.save_image(img, os.path.join(evalPath,saveName))

    print("Epoch done: " + str(epoch))
