import os
from src.lucyd import LUCYD
from src.train import train
from tqdm import tqdm
import torch
from torch.utils import data
import numpy as np
import tifffile
import glob
import logging
import datetime
from csbdeep.utils import normalize as _csbdeep_normalize


class ImageLoader(data.Dataset):
    def __init__(self, gt, blur, depth):
        # gt and blur: torch tensors
        # depth: number of slices for forward pass

        self.crop_depth = depth
        self.crop_size = 64
        self.dimZ = gt.shape[1]
        self.dimY = gt.shape[2]
        self.dimX = gt.shape[3]

        self.gt = gt
        self.blur = blur

        self.len = 128

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        im_index = np.random.randint(0, self.gt.shape[0])

        z_index = 0
        x_index = np.random.randint(0, self.dimX - self.crop_size)
        y_index = np.random.randint(0, self.dimY - self.crop_size)

        blur = self.blur[im_index][z_index:(z_index+self.crop_depth),x_index:(x_index+self.crop_size),y_index:(y_index+self.crop_size)]
        gt = self.gt[im_index][z_index:(z_index+self.crop_depth),x_index:(x_index+self.crop_size),y_index:(y_index+self.crop_size)]

        blur = torch.unsqueeze(blur, dim=0)
        gt = torch.unsqueeze(gt, dim=0)

        return blur, gt

def read_data():
    base_folder = '/mnt/NewDisk/PycharmProjects/pinhole_deblurring/data/Confocal_Pinhole/train/'
    train_folders = ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'v2_p06', 'v2_p10', 'v2_p20',
                   'v2_p30', 'v2_p40', 'v2_p50', 'v2_p60']
    clean_folder = 'clean'

    # calculate number of images
    num_images = 0
    small_test_dataset = False
    training_list = []
    for folder in train_folders:

        imgs = glob.glob(os.path.join(base_folder, folder, '*.tif'))
        for img in imgs:
            if small_test_dataset and num_images == 8:
                break
            # find matching gt
            gt = os.path.join(base_folder, clean_folder, os.path.basename(img))
            if not os.path.exists(gt):
                continue
            training_list.append((img, gt))
            num_images += 1

    # load one to get the size
    x_sample = tifffile.imread(training_list[0][0], maxworkers=6)
    y_sample = tifffile.imread(training_list[0][1], maxworkers=6)
    assert x_sample.shape == y_sample.shape
    dimZ, dimY, dimX = x_sample.shape

    blur_data = np.zeros((num_images,dimZ,dimY,dimX))
    gt_data = np.zeros((num_images,dimZ,dimY,dimX))

    for i in tqdm(range(num_images)):
        x_blur = tifffile.imread(training_list[i][0], maxworkers=6)
        x_gt = tifffile.imread(training_list[i][1], maxworkers=6)
        blur_data[i-1] = x_blur
        gt_data[i-1] = x_gt

    logging.info(f'Loaded {num_images} images')

    # use csbdeep for normalisation
    logging.info(f'Normalizing data using csbdeep')
    for i in tqdm(range(len(blur_data))):
        blur_data[i] = _csbdeep_normalize(blur_data[i])
        gt_data[i] = _csbdeep_normalize(gt_data[i])
    logging.info(f'Data normalized using csbdeep')

    logging.info(f'Converting to torch tensors')
    blur_data = torch.from_numpy(blur_data)
    gt_data = torch.from_numpy(gt_data)
    logging.info(f'Data converted to torch tensors')

    return blur_data, gt_data


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info('Reading data')
imgs, gts = read_data()
logging.info('Data read')

val_split = 0.2
depth = 32
batch_size = 4

val_size = int(val_split * len(imgs))
logging.info(f'Validation size: {val_size}')
logging.info(f'Training size: {len(imgs) - val_size}')
imgs_train, imgs_val = imgs[val_size:], imgs[:val_size]
gts_train, gts_val = gts[val_size:], gts[:val_size]
train_dataloader= data.DataLoader(ImageLoader(gts_train, imgs_train, depth), batch_size=batch_size, shuffle=True)
val_dataloader = data.DataLoader(ImageLoader(gts_val, imgs_val, depth), batch_size=batch_size, shuffle=True)
logging.info('Data loaded')


model = LUCYD(num_res=1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

logging.info('Training src')
model = train(model, train_dataloader, val_dataloader)
logging.info('Model trained')

now = datetime.datetime.now()
timestamp = now.strftime("%Y_%m_%d_%H_%M")
string_norm = 'percentile'
torch.save(model.state_dict(), f'./model_{timestamp}_{string_norm}.pt')
logging.info('Model saved')