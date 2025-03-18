from cellpose_omni import models, core, io
from skimage.measure import block_reduce
from tqdm import tqdm
from pathlib import Path
from cellpose_omni import io
import omnipose
import os.path
import shutil
from skimage.transform import resize
import glob
import itk
import logging
import numpy as np
import datetime
from cellpose_omni import models

use_GPU = core.use_gpu()
omnidir = Path(omnipose.__file__).parent.parent

io.logger_setup()

path_to_base_folder = '/mnt/work/data/CM_Seg_Paper_24/data_v2'
output_folder = '/mnt/work/data/CM_Seg_Paper_24/cellpose'
output_folder_train = os.path.join(output_folder, 'train')
output_folder_val = os.path.join(output_folder, 'val')
output_folder_test = os.path.join(output_folder, 'test')

namefilter = ''

all_test_imgs_nrrds = glob.glob(os.path.join(path_to_base_folder, "*", "test", 'imgs_norm', f'*{namefilter}*.nrrd'),
                                    recursive=True)

output_folder = '/mnt/work/data/CM_Seg_Paper_24/results'
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
output_folder = os.path.join(output_folder, f'omnipose_test_{timestamp}')
if os.path.exists(output_folder):
    logging.info(f"deleting {output_folder}")
    shutil.rmtree(output_folder)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


model_path = "/mnt/work/data/CM_Seg_Paper_24/omnipose/train/models/cellpose_residual_on_style_on_concatenation_off_omni_abstract_nclasses_3_nchan_1_dim_3_train_2025_03_14_23_02_13.979890_epoch_1000"
dim = 3
nclasses = 3 # flow + dist + boundary
nchan = 1
omni = 1
rescale = False
diam_mean = 0
use_GPU = 1 # Most people do not have enough VRAM to run on GPU... 24GB not enough for this image, need nearly 48GB
# model = models.CellposeModel(gpu=use_GPU, model_type=model_name, net_avg=False,
                             # diam_mean=diam_mean, nclasses=nclasses, dim=dim, nchan=nchan)

model = models.CellposeModel(gpu=use_GPU, pretrained_model=model_path, net_avg=False,
                             diam_mean=diam_mean, nclasses=nclasses, dim=dim, nchan=nchan)
model_cpu = models.CellposeModel(gpu=0, pretrained_model=model_path, net_avg=False,
                             diam_mean=diam_mean, nclasses=nclasses, dim=dim, nchan=nchan)
import torch
torch.cuda.empty_cache() # pretty much default parameters -- i tried tuning on val, but default seems to be ok
mask_threshold = -1 # -5 #usually this is -1
flow_threshold = 0
diam_threshold = 12
net_avg = False
cluster = False
verbose = 1
tile = True
chans = None
compute_masks = 1
resample=False
rescale=None
omni=True
flow_factor = 10 # multiple to increase flow magnitude, useful in 3D; going higher did not help in our tests
transparency = True
min_size = 15#10000


# splitting the images into batches helps manage VRAM use so that memory can get properly released
# here we have just one image, but most people will have several to process
for path_to_3d_stack in tqdm(all_test_imgs_nrrds):
    img = itk.GetArrayFromImage(itk.imread(path_to_3d_stack))
    original_shape = img.shape

    # downsample 3x
    img = block_reduce(img, (3, 3, 3), np.mean)
    try:
        prediction, flows, _ = model.eval(img,
                                                 channels=chans,
                                                 rescale=rescale,
                                                 mask_threshold=mask_threshold,
                                                 net_avg=net_avg,
                                                 transparency=transparency,
                                                 flow_threshold=flow_threshold,
                                                 omni=omni,
                                                 resample=resample,
                                                 verbose=verbose,
                                                 diam_threshold=diam_threshold,
                                                 cluster=cluster,
                                                 niter=100,
                                                 tile=tile,
                                                 min_size = min_size,
                                                 compute_masks=compute_masks,
                                                 flow_factor=flow_factor)
    except Exception as e:
        # cpu, some images are too large for GPU
        print(f"Error in {path_to_3d_stack}: {e}")
        print("Trying CPU")
        prediction , flows, _ = model_cpu.eval(img,
                                                 channels=chans,
                                                 rescale=rescale,
                                                 mask_threshold=mask_threshold,
                                                 net_avg=net_avg,
                                                 transparency=transparency,
                                                 flow_threshold=flow_threshold,
                                                 omni=omni,
                                                 resample=resample,
                                                 verbose=verbose,
                                                 diam_threshold=diam_threshold,
                                                 cluster=cluster,
                                                 niter=100,
                                                 tile=tile,
                                                 min_size = min_size,
                                                 compute_masks=compute_masks,
                                                 flow_factor=flow_factor)

    # resize up to original size
    prediction = resize(prediction, original_shape, order=0)

    filename_no_extension = os.path.splitext(os.path.basename(path_to_3d_stack))[0]
    fileext_out = 'nrrd'
    output_path = os.path.join(output_folder, f'{filename_no_extension}.{fileext_out}')
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    itk.imwrite(itk.GetImageFromArray(prediction), output_path, compression=True)