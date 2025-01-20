import logging
import glob
import os
import shutil
import warnings
from multiprocessing import Pool
from os.path import join
from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p
from skimage.morphology import dilation
from skimage.segmentation import find_boundaries
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw
import time
import numpy as np
import itk


def tic():
    return time.time()


def toc(t, prefixStr=""):
    duration = (time.time() - t)
    print(f"{prefixStr} Elapsed time: {duration:2f} s")
    return duration


def ITKReaderToNumpy(filePath, swapAxis=False, verbose=True):
    print("Loading with ITK: %s" % filePath)
    t = tic()
    data = itk.GetArrayFromImage(itk.imread(filePath))
    if verbose:
        print(data.shape)
    if len(data.shape) != 2:
        if swapAxis:
            data = np.swapaxes(data, 0, 2).copy()

    toc(t)
    print("\n")
    return data


def prepare_masks(path_to_instance_segmentation, path_to_image, output_path_img, output_path_mask):
    wga = ITKReaderToNumpy(path_to_image)
    segmentation = ITKReaderToNumpy(path_to_instance_segmentation)

    if wga.shape[2] < wga.shape[0]:
        warnings.warn(f'wga.shape[0] < wga.shape[2] for {path_to_image} -- correct??.')
        print(f'wga.shape[0] < wga.shape[2] for {path_to_image}.')
        print(f'wga.shape: {wga.shape}')
        print(f'segmentation.shape: {segmentation.shape}')

    if wga.shape != segmentation.shape:
        warnings.warn(f'Shape mismatch between {path_to_image} and {path_to_instance_segmentation}.')
        print(f'Shape mismatch between {path_to_image} and {path_to_instance_segmentation}.')
        print(f'wga.shape: {wga.shape}')
        print(f'segmentation.shape: {segmentation.shape}')
        print(f'wga.dtype: {wga.dtype}')
        print(f'segmentation.dtype: {segmentation.dtype}')
        return

    number_dilations = 2
    dilated_segmentation = segmentation.copy()
    for i in range(number_dilations):
        dilated_segmentation = dilation(dilated_segmentation)

    boundaries = find_boundaries(dilated_segmentation, connectivity=3, mode='thick')

    combined_mask = 2 * np.ones(shape=(segmentation.shape), dtype=np.uint8)
    combined_mask[segmentation > 0] = 0
    combined_mask[boundaries > 0] = 1

    itk.imwrite(itk.GetImageFromArray(wga), output_path_img)
    itk.imwrite(itk.GetImageFromArray(combined_mask), output_path_mask)


if __name__ == '__main__':
    skipIfFileExists = False
    path_to_base_folder = '/mnt/work/data/CM_Seg_Paper_24/data_v2/'
    number_of_processes = 12
    task_id = 821
    task_name = "WGA_Bnd"

    foldername = "Dataset%03.0d_%s" % (task_id, task_name)

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%d/%m/%Y %H:%M:%S')

    # setting up nnU-Net folders
    out_base = join(nnUNet_raw, foldername)

    # delete existing dataset folder, if exists
    if os.path.exists(out_base):
        shutil.rmtree(out_base)

    path_img_dir = join(out_base, "imagesTr")
    path_label_dir = join(out_base, "labelsTr")
    path_img_dir_test = join(out_base, "imagesTs")
    path_label_dir_test = join(out_base, "labelsTs")
    maybe_mkdir_p(path_img_dir)
    maybe_mkdir_p(path_label_dir)
    maybe_mkdir_p(path_img_dir_test)
    maybe_mkdir_p(path_label_dir_test)

    jobList = []

    all_train_imgs_nrrds = glob.glob(join(path_to_base_folder, "*", "train", 'imgs_norm', '*.nrrd'), recursive=True)

    for path_to_image in all_train_imgs_nrrds:
        stackname = os.path.splitext(os.path.basename(path_to_image))[0]
        path_to_instance_seg = path_to_image.replace('imgs_norm', 'labels')
        assert os.path.exists(path_to_instance_seg), f'Could not find {path_to_instance_seg}'
        output_path_img = join(path_img_dir, stackname + '_0000.nii.gz')
        output_path_seg = join(path_label_dir, stackname + '.nii.gz')
        jobList.append([path_to_instance_seg, path_to_image, output_path_img, output_path_seg])

    all_test_imgs_nrrds = glob.glob(join(path_to_base_folder, "*", "test", 'imgs_norm', '*.nrrd'), recursive=True)
    for path_to_image in all_test_imgs_nrrds:
        stackname = os.path.splitext(os.path.basename(path_to_image))[0]
        path_to_instance_seg = path_to_image.replace('imgs_norm', 'labels')
        assert os.path.exists(path_to_instance_seg), f'Could not find {path_to_instance_seg}'
        output_path_img = join(path_img_dir_test, stackname + '_0000.nii.gz')
        output_path_seg = join(path_label_dir_test, stackname + '.nii.gz')
        jobList.append([path_to_instance_seg, path_to_image, output_path_img, output_path_seg])


    if number_of_processes > 1:
        with Pool(processes=number_of_processes) as pool:
            pool.starmap(prepare_masks, jobList)
    else:
        for job in jobList:
            prepare_masks(*job)

    generate_dataset_json(out_base,
                          channel_names={0: 'wga'},
                          labels={
                              'background': 0,
                              'bnd': 1,
                              'ignore': 2
                          },
                          num_training_cases=len(all_train_imgs_nrrds),
                          file_ending='.nii.gz',
                          license='',
                          reference='',
                          dataset_release='0.01')
