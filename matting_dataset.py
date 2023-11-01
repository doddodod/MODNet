from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import cv2
from glob import glob
import torch
from torchvision.transforms import functional as F
from torchvision import transforms
from PIL import Image

class MattingDataset(Dataset):
    def __init__(self,
                 dataset_root_dir='dataset/UGD-12k', #dataset/PPM-100/train or dataset/UGD-12k 
                 transform=None):
        image_path = dataset_root_dir + '/image/*' # fg if PPM, image if UGD
        matte_path = dataset_root_dir + '/alpha/*'         
        image_file_name_list = glob(image_path)
        matte_file_name_list = glob(matte_path)

        self.image_file_name_list = sorted(image_file_name_list)
        self.matte_file_name_list = sorted(matte_file_name_list)
        for img, mat in zip(self.image_file_name_list, self.matte_file_name_list):
            img_name = img.split('/')[-1].split('.')[0]  # Get the name without extension 
            mat_name = mat.split('/')[-1].split('.')[0] 
            assert img_name == mat_name

        self.transform = transform

    def __len__(self):
        return len(self.image_file_name_list)

    def __getitem__(self, index):
        image_file_name = self.image_file_name_list[index]
        matte_file_name = self.matte_file_name_list[index]

        image = Image.open(image_file_name)
        matte = Image.open(matte_file_name)
        # matte = matte.convert('RGB')
        trimap = self.gen_trimap(matte)

        data = {'image': image, 'trimap': trimap, 'gt_matte': matte}

        if self.transform:
            data = self.transform(data)
        return data

    @staticmethod
    def gen_trimap(matte):
        """
        get trimap by matte
        """
        matte = np.array(matte)
        k_size = random.choice(range(2, 5))
        iterations = np.random.randint(5, 15)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (k_size, k_size))
        dilated = cv2.dilate(matte, kernel, iterations=iterations)
        eroded = cv2.erode(matte, kernel, iterations=iterations)

        trimap = np.zeros(matte.shape)
        trimap.fill(128)
        trimap[eroded > 254.5] = 255
        trimap[dilated < 0.5] = 0
        trimap = Image.fromarray(np.uint8(trimap))
        return trimap


class Rescale(object):
    """Rescale the image in a sample to a given size.
    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, trimap, gt_matte = sample['image'], sample['trimap'], sample['gt_matte']

        # w, h = image.size
        # if h > w:
        #     new_h, new_w = self.output_size * h / w, self.output_size
        # else:
        #     new_h, new_w = self.output_size, self.output_size * w / h

        # new_h, new_w = int(new_h), int(new_w)
        new_h, new_w = int(self.output_size), int(self.output_size)

        new_img = F.resize(image, (new_h, new_w))
        new_trimap = F.resize(trimap, (new_h, new_w))
        new_gt_matte = F.resize(gt_matte, (new_h, new_w))

        return {'image': new_img, 'trimap': new_trimap, 'gt_matte': new_gt_matte}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, trimap, gt_matte = sample['image'], sample['trimap'], sample['gt_matte']
        image = F.pil_to_tensor(image)
        trimap = F.pil_to_tensor(trimap)
        gt_matte = F.pil_to_tensor(gt_matte)
        return {'image': image,
                'trimap': trimap,
                'gt_matte': gt_matte}


class ConvertImageDtype(object):
    def __call__(self, sample):
        image, trimap, gt_matte = sample['image'], sample['trimap'], sample['gt_matte']
        image = F.convert_image_dtype(image, torch.float)
        trimap = F.convert_image_dtype(trimap, torch.float)
        gt_matte = F.convert_image_dtype(gt_matte, torch.float)

        return {'image': image, 'trimap': trimap, 'gt_matte': gt_matte}


class Normalize(object):
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, sample):
        image, trimap, gt_matte = sample['image'], sample['trimap'], sample['gt_matte']
        image = image.type(torch.FloatTensor)
        image = F.normalize(image, self.mean, self.std, self.inplace)
        sample['image'] = image

        sample['trimap'] = trimap / 255 
        return sample


class ToTrainArray(object):
    def __call__(self, sample):
        return [sample['image'], sample['trimap'], sample['gt_matte']]


if __name__ == '__main__':

    # test MattingDataset.gen_trimap
    matte = Image.open('dataset/PPM-100/train/alpha/14299313536_ea3e61076c_o.jpg')
    matte = matte.convert('RGB')
    trimap = MattingDataset.gen_trimap(matte)
    trimap.save('test_trimap.png')

    # test MattingDataset
    transform = transforms.Compose([
        Rescale(512),
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    mattingDataset = MattingDataset(transform=transform)

    import matplotlib.pyplot as plt

    fig = plt.figure()

    for i in range(len(mattingDataset)):
        sample = mattingDataset[i]
        print(mattingDataset.image_file_name_list[i])
        # print(sample)
        print(i, sample['image'].shape, sample['trimap'].shape, sample['gt_matte'].shape)

        # break

        ax = plt.subplot(4, 3, 3 * i + 1)
        plt.tight_layout()
        ax.set_title('image #{}'.format(i))
        ax.axis('off')
        img = transforms.ToPILImage()(sample['image'])
        plt.imshow(img)

        ax = plt.subplot(4, 3, 3 * i + 2)
        plt.tight_layout()
        ax.set_title('gt_matte #{}'.format(i))
        ax.axis('off')
        img = transforms.ToPILImage()(sample['gt_matte'])
        plt.imshow(img)

        ax = plt.subplot(4, 3, 3 * i + 3)
        plt.tight_layout()
        ax.set_title('trimap #{}'.format(i))
        ax.axis('off')
        img = transforms.ToPILImage()(sample['trimap'])
        plt.imshow(img)

        if i == 3:
            plt.show()
            break