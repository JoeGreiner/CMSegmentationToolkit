import os
import shutil

import numpy as np
import logging

import pandas as pd


def check_file_exists(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'File not found: {file_path}')


def backup_if_exists(file_path):
    dirname_output = os.path.dirname(file_path)
    if not os.path.exists(dirname_output):
        os.makedirs(dirname_output)

    backup_folder = os.path.join(dirname_output, 'backup')
    if not os.path.exists(backup_folder):
        os.makedirs(backup_folder)
    timestamp = pd.Timestamp.now().strftime('%Y_%m_%d-%H_%M_%S')
    old_filename = os.path.splitext(os.path.basename(file_path))[0]
    file_ext = os.path.splitext(file_path)[1]
    backup_file = os.path.join(backup_folder, f'{old_filename}_{timestamp}{file_ext}')
    if os.path.exists(file_path):
        shutil.copy(file_path, backup_file)
        print(f'Backup of {file_path} saved to {backup_file}')


def cast_to_uint32_if_float(pred):
    if pred.dtype == np.float32 or pred.dtype == np.float64:
        logging.info('Casting float to uint32')
        # check if fits in uint32
        max_val = pred.max()
        min_val = pred.min()
        assert max_val < 2 ** 32, f'Image exceeds uint32: {max_val}'
        assert min_val >= 0, f'Image has negative values: {min_val}'
        return pred.astype(np.uint32)
    logging.info('No casting needed, dtype is not float')
    return pred


def match_pred_and_gt_shape(pred, gt, wga=None, crop_to_fit=False):
    if wga is not None:
        assert gt.shape == wga.shape, f'Shape mismatch between wga and gt: {gt.shape} // {wga.shape}'

    # check shapes
    if pred.shape != gt.shape:
        # put smallest and largest axis at the same position
        logging.warning(f'Shape mismatch: {pred.shape} // {gt.shape}')
        # this may need swapping though ...
        argmin_shape_gt = np.argmin(gt.shape)
        argmin_shape_pred = np.argmin(pred.shape)
        logging.warning(f'Argmin shapes: {argmin_shape_gt} // {argmin_shape_pred}')
        if argmin_shape_pred != argmin_shape_gt:
            logging.warning(f'Swapping axes indices: {argmin_shape_gt} // {argmin_shape_pred}')
            gt = np.swapaxes(gt, argmin_shape_gt, argmin_shape_pred)
            if wga is not None:
                wga = np.swapaxes(wga, argmin_shape_gt, argmin_shape_pred)
        # do the same with argmax
        argmax_shape_gt = np.argmax(gt.shape)
        argmax_shape_pred = np.argmax(pred.shape)
        logging.warning(f'Argmax shapes: {argmax_shape_gt} // {argmax_shape_pred}')
        if argmax_shape_pred != argmax_shape_gt:
            logging.warning(f'Swapping axes indices: {argmax_shape_gt} // {argmax_shape_pred}')
            gt = np.swapaxes(gt, argmax_shape_gt, argmax_shape_pred)
            if wga is not None:
                wga = np.swapaxes(wga, argmax_shape_gt, argmax_shape_pred)

        # now, crop if needed
        if pred.shape != gt.shape:
            if crop_to_fit:
                logging.warning(f'Cropping to fit')
                logging.warning(f'Old shape: {gt.shape}')
                gt = gt[:pred.shape[0], :pred.shape[1], :pred.shape[2]]
                if wga is not None:
                    wga = wga[:pred.shape[0], :pred.shape[1], :pred.shape[2]]
                logging.warning(f'New shape: {gt.shape}')
                logging.warning(f'pred shape: {pred.shape}')
                if pred.shape != gt.shape:
                    logging.warning(f'Shape mismatch after crop: {pred.shape} // {gt.shape}')
                    logging.warning(f'ALSO MATCHING PREDICTION SHAPE TO GT (unusual)')
                    logging.warning(f'ALSO MATCHING PREDICTION SHAPE TO GT (unusual)')
                    logging.warning(f'ALSO MATCHING PREDICTION SHAPE TO GT (unusual)')
                    pred = pred[:gt.shape[0], :gt.shape[1], :gt.shape[2]]
                    logging.warning(f'pred shape: {pred.shape}')
            else:
                raise ValueError(f'Shape mismatch: {pred.shape} // {gt.shape}')
    else:
        logging.info(f'Shape match: {pred.shape} // {gt.shape}')
        print(f'Shape match: {pred.shape} // {gt.shape}')
    if wga is not None:
        return pred.copy(), gt.copy(), wga.copy()
    return pred.copy(), gt.copy()


def align_array_shape(array_to_match_to, array_to_match):
    # given array_to_match_to with the shape (z, y, x) and array_to_match with the shape (x, y, z), find the matching
    # dimension permutation so that the shapes match

    assert array_to_match_to.ndim == array_to_match.ndim, f'ndim mismatch: {array_to_match_to.ndim} // {array_to_match.ndim}'

    shape_to_match_to = array_to_match_to.shape
    shape_to_match = array_to_match.shape

    # check if shapes match
    if shape_to_match_to == shape_to_match:
        return array_to_match

    # check if a simple 0 -- 2 axis swap would fix things
    if shape_to_match_to == (shape_to_match[2], shape_to_match[1], shape_to_match[0]):
        return np.swapaxes(array_to_match, 0, 2).copy()

    permute_axis = []
    for dim_size_to_match_to in shape_to_match_to:
        for axis, dim_size_to_match in enumerate(shape_to_match):
            if dim_size_to_match_to == dim_size_to_match:
                # check if already in list
                if axis in permute_axis:
                    # take next axis
                    continue
                permute_axis.append(axis)
                break

    if len(permute_axis) != array_to_match.ndim:
        raise ValueError('Could not find matching permutation')

    return np.transpose(array_to_match, axes=permute_axis).copy()
