import glob
import shutil
import numpy as np
import itk
import h5py
import logging
from multiprocessing import Pool
import os

def process_stack(path_to_image, path_to_instance_seg, output_folder):
    stackname = os.path.splitext(os.path.basename(path_to_image))[0]
    assert os.path.exists(path_to_instance_seg), f'Could not find {path_to_instance_seg}'
    assert os.path.exists(path_to_image), f'Could not find {path_to_image}'

    img = itk.GetArrayFromImage(itk.imread(path_to_image))
    label = itk.GetArrayFromImage(itk.imread(path_to_instance_seg))

    assert img.shape == label.shape, f'Shape mismatch between {path_to_image} and {path_to_instance_seg}'
    logging.info(f'Processing {stackname}, shape: {img.shape} (should be zyx)')

    with h5py.File(os.path.join(output_folder, stackname + '.h5'), 'w') as f:
        f.create_dataset('raw', data=img, dtype='uint16', compression='lzf', chunks=True)
        f.create_dataset('label', data=label, dtype='uint32', compression='lzf', chunks=True)

def process_path_img_list(path_img_list, output_folder, number_processes=6):
    logging.info(f'Processing {len(path_img_list)} images; writing to {output_folder}')
    job_list = []
    for path_to_image in path_img_list:
        path_to_instance_seg = path_to_image.replace('imgs_norm', 'labels')
        assert os.path.exists(path_to_instance_seg), f'Could not find {path_to_instance_seg}'
        job_list.append((path_to_image, path_to_instance_seg, output_folder))

    if number_processes > 1:
        with Pool(number_processes) as p:
            p.starmap(process_stack, job_list)
    else:
        for job in job_list:
            process_stack(*job)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    path_to_base_folder = '/mnt/work/data/CM_Seg_Paper_24/data_v2/'
    # base folder: subfolder: human, myocardial_infarction, slice_culture, species
    # subsubfolders: test, train
    # subsubsubfolders: imgs_norm, labels

    output_folder_base = "/mnt/NewDisk/data/wolny_unet/run_2024_09_28_v2"

    output_folder_train = os.path.join(output_folder_base, 'train')
    output_folder_val = os.path.join(output_folder_base, 'val')
    output_folder_test = os.path.join(output_folder_base, 'test')
    logging.info(f'Processing data from {path_to_base_folder}')
    logging.info(f'Output folders: {output_folder_train}, {output_folder_val}, {output_folder_test}')

    val_fraction = 0.2
    logging.info(f'Validation fraction: {val_fraction}')

    np.random.seed(42)

    assert os.path.exists(path_to_base_folder), f'Could not find {path_to_base_folder}'
    for folder in [output_folder_train, output_folder_val, output_folder_test]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)

    all_train_and_val_imgs_nrrds = glob.glob(os.path.join(path_to_base_folder, "*", "train", 'imgs_norm', '*.nrrd'),
                                             recursive=True)
    all_test_imgs_nrrds = glob.glob(os.path.join(path_to_base_folder, "*", "test", 'imgs_norm', '*.nrrd'),
                                    recursive=True)

    assert len(all_train_and_val_imgs_nrrds) > 0, 'Could not find any training images'
    assert len(all_test_imgs_nrrds) > 0, 'Could not find any test images'

    np.random.shuffle(all_train_and_val_imgs_nrrds)
    n_val = int(val_fraction * len(all_train_and_val_imgs_nrrds))
    all_train_imgs_nrrds = all_train_and_val_imgs_nrrds[n_val:]
    all_val_imgs_nrrds = all_train_and_val_imgs_nrrds[:n_val]

    logging.info(f'{len(all_train_imgs_nrrds)} training images')
    logging.info(f'{len(all_val_imgs_nrrds)} validation images')
    logging.info(f'{len(all_test_imgs_nrrds)} test images')

    process_path_img_list(all_train_imgs_nrrds, output_folder_train)
    process_path_img_list(all_val_imgs_nrrds, output_folder_val)
    process_path_img_list(all_test_imgs_nrrds, output_folder_test)
