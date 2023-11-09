import torch
import numpy as np
import torch.nn as nn
from glob import glob
from PIL import Image
from infer import predit_matte
from scipy.ndimage import gaussian_filter
from skimage.measure import label, regionprops
from scipy.ndimage import distance_transform_edt

#Change it to the model type using:  
from src.models.modnet_old import MODNet #baseline MODNet
#from src.models.modnet import MODNet #ViTae 
#from src.models.modnet_tfi import MODNet #TFI


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

def cal_sad(pred, gt):
    diff = pred - gt
    diff = np.abs(diff)
    sad = np.sum(diff)
    return sad/10000

def cal_grad(pred, gt):
    pd_x = gaussian_filter(pred, sigma=1.4, order=[1, 0], output=np.float32)
    pd_y = gaussian_filter(pred, sigma=1.4, order=[0, 1], output=np.float32)
    gt_x = gaussian_filter(gt, sigma=1.4, order=[1, 0], output=np.float32)
    gt_y = gaussian_filter(gt, sigma=1.4, order=[0, 1], output=np.float32)
    pd_mag = np.sqrt(pd_x**2 + pd_y**2)
    gt_mag = np.sqrt(gt_x**2 + gt_y**2)
    error_map = np.square(pd_mag - gt_mag)
    loss = np.sum(error_map) / 10
    return loss

def cal_conn(pred, gt, step=0.1):
    h, w = pred.shape
    thresh_steps = np.arange(0, 1.1, step)
    l_map = -1 * np.ones((h, w), dtype=np.float32)
    for i in range(1, thresh_steps.size):
        pd_th = pred >= thresh_steps[i]
        gt_th = gt >= thresh_steps[i]
        label_image = label(pd_th & gt_th, connectivity=1)
        cc = regionprops(label_image)
        size_vec = np.array([c.area for c in cc])
        if len(size_vec) == 0:
            continue
        max_id = np.argmax(size_vec)
        coords = cc[max_id].coords
        omega = np.zeros((h, w), dtype=np.float32)
        omega[coords[:, 0], coords[:, 1]] = 1
        flag = (l_map == -1) & (omega == 0)
        l_map[flag == 1] = thresh_steps[i-1]
        dist_maps = distance_transform_edt(omega==0)
        dist_maps = dist_maps / dist_maps.max()
    l_map[l_map == -1] = 1
    d_pd = pred - l_map
    d_gt = gt - l_map
    phi_pd = 1 - d_pd * (d_pd >= 0.15).astype(np.float32)
    phi_gt = 1 - d_gt * (d_gt >= 0.15).astype(np.float32)
    loss = np.sum(np.abs(phi_pd - phi_gt)) / 10000
    return loss 

def eval(modnet: MODNet, dataset):
    mse = total_mse = 0.0
    mad = total_mad = 0.0
    sad = total_sad = 0.0
    grad = total_grad = 0.0
    conn = total_conn = 0.0
    cnt = 0

    for im_pth, mt_pth in zip(dataset[0], dataset[1]):
        im = Image.open(im_pth)
        pd_matte = predit_matte(modnet, im)

        gt_matte = Image.open(mt_pth)
        gt_matte = np.asarray(gt_matte) / 255

        total_mse += cal_mse(pd_matte, gt_matte)
        total_mad += cal_mad(pd_matte, gt_matte)
        total_sad += cal_sad(pd_matte, gt_matte)
        total_grad += cal_grad(pd_matte, gt_matte)
        total_conn += cal_conn(pd_matte, gt_matte)

        cnt += 1
    if cnt > 0:
        mse = total_mse / cnt
        mad = total_mad / cnt
        sad = total_sad / cnt
        grad = total_grad / cnt
        conn = total_conn / cnt

    return mse, mad, sad, grad, conn

def call_eval(modnet): 
    data = 'dataset/UGD-12k' 
    image_path = data + '/eval/image/*' 
    matte_path = data + '/eval/alpha/*'
    print(data) 
    dataset = load_eval_dataset(image_path, matte_path) 
    mse, mad, sad, grad, conn = eval(modnet, dataset)
    print(f'mse: {mse:6f}, mad: {mad:6f}, sad: {sad:6f}, grad: {grad:6f}, conn: {conn:6f}') 
    return None 
    

if __name__ == '__main__':
    # create MODNet and load the pre-trained ckpt
    modnet = MODNet(backbone_pretrained=False)
    modnet = nn.DataParallel(modnet)
    
    #ckp_pth = 'pretrained/UGD-12k_trained_model.pth' 
    #ckp_pth = 'pretrained/UGD-12k_on_pretrained_model.pth' 
    #ckp_pth = 'pretrained/vitae_epoch23.pth'
    ckp_pth = 'pretrained/vitae_epoch21.pth' 
    
    # data = 'dataset/PPM-100' 
    # image_path = data + '/val/fg/*'
    # matte_path = data + '/val/alpha/*'
    
    data = 'dataset/UGD-12k' 
    image_path = data + '/eval/image/*' 
    matte_path = data + '/eval/alpha/*' 
    
    print(ckp_pth)
    print(data) 
    
    if torch.cuda.is_available():
        modnet = modnet.cuda()
        weights = torch.load(ckp_pth)
    else:
        weights = torch.load(ckp_pth, map_location=torch.device('cpu'))
    modnet.load_state_dict(weights)
    
    dataset = load_eval_dataset(image_path, matte_path) 
    mse, mad, sad, grad, conn = eval(modnet, dataset)
    print(f'mse: {mse:6f}, mad: {mad:6f}, sad: {sad:6f}, grad: {grad:6f}, conn: {conn:6f}') 