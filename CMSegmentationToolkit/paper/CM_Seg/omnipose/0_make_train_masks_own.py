import os
import glob
import numpy as np
import tifffile
import tqdm
import logging
import itk
import shutil

from skimage.measure import block_reduce

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    path_to_base_folder = '/mnt/work/data/CM_Seg_Paper_24/data_v2'
    output_folder = '/mnt/work/data/CM_Seg_Paper_24/omnipose'
    output_folder_train = os.path.join(output_folder, 'train')
    output_folder_val = os.path.join(output_folder, 'val')
    output_folder_test = os.path.join(output_folder, 'test')

    for folder in [output_folder_train, output_folder_val, output_folder_test]:
        if os.path.exists(folder):
            logging.info(f"deleting {folder}")
            shutil.rmtree(folder)

        if not os.path.exists(folder):
            os.makedirs(folder)

    all_train_and_val_imgs_nrrds = glob.glob(os.path.join(path_to_base_folder, "*", "train", 'imgs_norm', '*.nrrd'),
                                             recursive=True)
    all_test_imgs_nrrds = glob.glob(os.path.join(path_to_base_folder, "*", "test", 'imgs_norm', '*.nrrd'),
                                    recursive=True)

    assert len(all_train_and_val_imgs_nrrds) > 0, 'Could not find any training images'
    assert len(all_test_imgs_nrrds) > 0, 'Could not find any test images'

    # seed 42
    np.random.seed(42)
    val_fraction = 0.2
    np.random.shuffle(all_train_and_val_imgs_nrrds)
    n_val = int(val_fraction * len(all_train_and_val_imgs_nrrds))
    all_train_imgs_nrrds = all_train_and_val_imgs_nrrds[n_val:]
    all_val_imgs_nrrds = all_train_and_val_imgs_nrrds[:n_val]

    logging.info(f'{len(all_train_imgs_nrrds)} training images')
    logging.info(f'{len(all_val_imgs_nrrds)} validation images')
    logging.info(f'{len(all_test_imgs_nrrds)} test images')

    for img_list, output_folder in zip([all_train_imgs_nrrds, all_val_imgs_nrrds, all_test_imgs_nrrds],
                                       [output_folder_train, output_folder_val, output_folder_test]):
        for img_path in tqdm.tqdm(img_list):
            path_to_instance_seg = img_path.replace('imgs_norm', 'labels')
            assert os.path.exists(path_to_instance_seg), f'Could not find {path_to_instance_seg}'

            img0 = itk.GetArrayFromImage(itk.imread(img_path))
            mask0 = itk.GetArrayFromImage(itk.imread(path_to_instance_seg))

            # documentation -- receptive field should be at least size of cell; 3x seems fine from tuning
            img0 = block_reduce(img0, (3, 3, 3), np.mean)
            mask0 = block_reduce(mask0, (3, 3, 3), np.max)

            assert img0.shape == mask0.shape, f"image shape {img0.shape} does not match mask shape {mask0.shape}"
            dimZ, dimY, dimX = img0.shape

            filename_no_ext = os.path.splitext(os.path.basename(img_path))[0]
            output_img = os.path.join(output_folder, f'{filename_no_ext}.tif')
            output_mask = os.path.join(output_folder,
                                       f'{filename_no_ext}_masks.tif')

            tifffile.imwrite(output_img, img0)
            tifffile.imwrite(output_mask, mask0)