import numpy as np
from glob import glob
from src.models.modnet import MODNet
from PIL import Image
from infer import predit_matte
import torch.nn as nn
import torch


def cal_mad(pred, gt):
    diff = pred - gt
    diff = np.abs(diff)
    mad = np.mean(diff)
    return mad


def cal_mse(pred, gt):
    diff = pred - gt
    diff = diff ** 2
    mse = np.mean(diff)
    return mse


def load_eval_dataset(image_path, matte_path): 
    image_file_name_list = glob(image_path)
    matte_file_name_list = glob(matte_path)

    # Sort the file lists
    image_file_name_list = sorted(image_file_name_list)
    matte_file_name_list = sorted(matte_file_name_list) 

    # Check if the sizes are the same and create a mask
    size_check_mask = [check_size(image, matte) for image, matte in zip(image_file_name_list, matte_file_name_list)]
    image_file_name_list = [image for i, image in enumerate(image_file_name_list) if size_check_mask[i]]
    matte_file_name_list = [matte for i, matte in enumerate(matte_file_name_list) if size_check_mask[i]] 

    return image_file_name_list, matte_file_name_list

def check_size(image_path, matte_path):
    image = Image.open(image_path)
    matte = Image.open(matte_path)
    return image.size == matte.size 

def eval(modnet: MODNet, dataset):
    mse = total_mse = 0.0
    mad = total_mad = 0.0
    cnt = 0

    for im_pth, mt_pth in zip(dataset[0], dataset[1]):
        im = Image.open(im_pth)
        pd_matte = predit_matte(modnet, im)

        gt_matte = Image.open(mt_pth)
        gt_matte = np.asarray(gt_matte) / 255

        total_mse += cal_mse(pd_matte, gt_matte)
        total_mad += cal_mad(pd_matte, gt_matte)

        cnt += 1
    if cnt > 0:
        mse = total_mse / cnt
        mad = total_mad / cnt

    return mse, mad


if __name__ == '__main__':
    # create MODNet and load the pre-trained ckpt
    modnet = MODNet(backbone_pretrained=False)
    modnet = nn.DataParallel(modnet)
    
    #pretrained/modnet_photographic_portrait_matting.ckpt or
    #pretrained/UGD-12k_trained_model.pth
    ckp_pth = 'pretrained/UGD-12k_trained_model.pth' 
    print(ckp_pth) 
    if torch.cuda.is_available():
        modnet = modnet.cuda()
        weights = torch.load(ckp_pth)
    else:
        weights = torch.load(ckp_pth, map_location=torch.device('cpu'))
    modnet.load_state_dict(weights)
    
    data = 'dataset/UGD-12k' # dataset/UGD-12k or dataset/PPM-100 
    image_path = data + '/eval/image/*' #/eval/image/* or /val/fg/*
    matte_path = data + '/eval/alpha/*' #/eval/alpha/* or /val/alpha/* 
    print(data)
    dataset = load_eval_dataset(image_path, matte_path) 
    
    mse, mad = eval(modnet, dataset)
    print(f'mse: {mse:6f}, mad: {mad:6f}')