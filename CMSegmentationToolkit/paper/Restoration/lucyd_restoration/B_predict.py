import os
from csbdeep.data import PercentileNormalizer
from src.lucyd import LUCYD
from tqdm import tqdm
import torch
from torch.utils import data
import numpy as np
import tifffile
import glob

def read_data():
    base_folder = '/mnt/NewDisk/PycharmProjects/pinhole_deblurring/data/Confocal_Pinhole/test/'
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
    stacknames = []

    for i in tqdm(range(num_images)):
        x_nuc = tifffile.imread(training_list[i][0], maxworkers=6)
        x_gt = tifffile.imread(training_list[i][1], maxworkers=6)
        blur_data[i] = x_nuc
        gt_data[i] = x_gt
        # stackname = folder + file
        filename = os.path.basename(training_list[i][0])
        foldername = os.path.basename(os.path.dirname(training_list[i][0]))
        stackname = foldername + '-' + filename
        stacknames.append(stackname)

    logging.info(f'Loaded {num_images} images')


    # use csbdeep for normalisation
    logging.info(f'Normalizing data using csbdeep')
    a = PercentileNormalizer()
    for i in tqdm(range(len(blur_data))):
        blur_data[i] = a.before(blur_data[i], axes='ZYX')
        gt_data[i] = a.before(gt_data[i], axes='ZYX')
    logging.info(f'Data normalized using csbdeep')


    logging.info(f'Converting to torch tensors')
    blur_data = torch.from_numpy(blur_data)
    gt_data = torch.from_numpy(gt_data)
    logging.info(f'Data converted to torch tensors')

    return blur_data, gt_data, stacknames


class ImageLoaderTest(data.Dataset):
    def __init__(self, gt, blur, stacknames):
        # gt and blur: torch tensors
        # depth: number of slices for forward pass

        self.dimZ = gt.shape[1]
        self.dimY = gt.shape[2]
        self.dimX = gt.shape[3]

        self.stacknames = stacknames

        self.gt = gt
        self.blur = blur

        number_samples_per_image = self.dimZ * self.dimY * self.dimX / (64 * 64 * 32)
        self.len = int(len(gt) * number_samples_per_image)
        logging.info(f'Number of samples: {self.len}')


    def __len__(self):
        return self.len

    def __getitem__(self, index):
        index_drawn = np.random.randint(0, len(self.gt))
        blur = self.blur[index_drawn]
        gt = self.gt[index_drawn]

        fz = 0
        fx = np.random.randint(0, self.dimX - 64)
        fy = np.random.randint(0, self.dimY - 64)
        tz = 32
        tx = fx + 64
        ty = fy + 64

        blur = blur[fz:tz, fx:tx, fy:ty]
        gt = gt[fz:tz, fx:tx, fy:ty]

        blur = torch.unsqueeze(blur, dim=0)
        gt = torch.unsqueeze(gt, dim=0)

        return blur, gt

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info('Reading data')
imgs, gts, stacknames = read_data()
logging.info('Data read')


model = LUCYD(num_res=1)
model.load_state_dict(torch.load('model_2024_09_29_15_34_percentile.pt'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()
logging.info('Model loaded')

prediction_folder = './predictions'
if not os.path.exists(prediction_folder):
    os.makedirs(prediction_folder)

# do gaussian blending image for sliding window predictions
stride = 0.5
batch_size = 4
window = (32, 64, 64)
sigma = 5
gauss_volume = np.zeros(window)
for z in range(window[0]):
    for y in range(window[1]):
        for x in range(window[2]):
            gauss_volume[z, y, x] = np.exp(-((z - window[0]//2)**2 + (y - window[1]//2)**2 + (x - window[2]//2)**2)/(2*sigma**2))


for index in range(len(imgs)):
    img_blurry = imgs[index]
    img_clear = gts[index]
    stackname = stacknames[index]

    # do sliding window interference
    counter_image = np.zeros(img_blurry.shape)
    sum_image = np.zeros(img_blurry.shape)

    batch_size_counter = 0
    current_x = []
    current_y = []
    current_img = []
    number_y_steps = int((img_blurry.shape[1] - window[1]) / (window[1] * stride)) + 1
    number_x_steps = int((img_blurry.shape[2] - window[2]) / (window[2] * stride)) + 1
    for y in tqdm(np.linspace(0, img_blurry.shape[1] - window[1], number_y_steps, dtype=int)):
        for x in np.linspace(0, img_blurry.shape[2] - window[2], number_x_steps, dtype=int):

            img = img_blurry[:, y:y + window[1], x:x + window[2]]
            img = torch.unsqueeze(torch.unsqueeze(img, dim=0), dim=0)

            img = img.float().to(device)

            current_x.append(x)
            current_y.append(y)
            current_img.append(img)
            batch_size_counter += 1

            if batch_size_counter == batch_size:
                batch_size_counter = 0
                img = torch.cat(current_img, dim=0)
                prediction, y_k, up = model(img)
                prediction = prediction.cpu().detach().numpy()
                for i in range(len(current_x)):
                    sum_image[:, current_y[i]:current_y[i] + window[1], current_x[i]:current_x[i] + window[2]] += np.squeeze(prediction[i])*gauss_volume
                    counter_image[:, current_y[i]:current_y[i] + window[1], current_x[i]:current_x[i] + window[2]] += gauss_volume
                current_x = []
                current_y = []
                current_img = []

    if len(current_img) > 0:
        img = torch.cat(current_img, dim=0)
        prediction, y_k, up = model(img)
        prediction = prediction.cpu().detach().numpy()
        for i in range(len(current_x)):
            sum_image[:, current_y[i]:current_y[i] + window[1], current_x[i]:current_x[i] + window[2]] += np.squeeze(prediction[i])*gauss_volume
            counter_image[:, current_y[i]:current_y[i] + window[1], current_x[i]:current_x[i] + window[2]] += gauss_volume

    counter_image[counter_image == 0] = 1
    prediction = sum_image / counter_image
    logging.info(f'Saving prediction for {stackname}')
    stackname_no_extension = stackname.replace('.tif', '')
    tifffile.imwrite(os.path.join(prediction_folder, f'{stackname_no_extension}.tif'), prediction)
    tifffile.imwrite(os.path.join(prediction_folder, f'{stackname_no_extension}_blurry.tif'), img_blurry.numpy())
    tifffile.imwrite(os.path.join(prediction_folder, f'{stackname_no_extension}_clear.tif'), img_clear.numpy())
