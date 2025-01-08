import os
import glob
import shutil

import itk
import pandas as pd
import numpy as np
from skimage.metrics import adapted_rand_error, variation_of_information
from stardist.matching import matching

from CMSegmentationToolkit.src.utils import match_pred_and_gt_shape


def evaluatate_single_image(gt_path: str, pred_path: str, crop_to_fit: bool = False):

    stackname = os.path.splitext(os.path.basename(pred_path))[0]
    stackname = stackname.replace("cell_segm_", "")
    if '_fx' in stackname:
        stackname = stackname.split('_fx')[0]
    if '_mc_0' in stackname:
        stackname = stackname.split('_mc_0')[0]

    if not os.path.exists(gt_path):
        print(f'GT file not found: {gt_path}')
        return

    pred = itk.GetArrayFromImage(itk.imread(pred_path))

    max_val = pred.max()
    min_val = pred.min()
    if max_val > 2 ** 32 - 1:
        print(f'Image exceeds uint32: {max_val}')
        return
    if min_val < 0:
        print(f'Image has negative values: {min_val}')
        return
    pred = pred.astype(np.uint32)
    gt = itk.GetArrayFromImage(itk.imread(gt_path))

    # inference issue on fabio's side, his suggestion was to crop the gt to fit the pred
    pred, gt = match_pred_and_gt_shape(pred, gt, crop_to_fit=crop_to_fit)
    assert pred.shape == gt.shape, f'Shape mismatch: {pred.shape} // {gt.shape}'

    # (split, merge) = voi(pred, gt)
    # print(f'{stackname} - split: {split:.2f} merge: {merge:.2f}')
    split_sk, merge_sk = variation_of_information(gt, pred, ignore_labels=[0, ])
    print(f'{stackname} - split_sk: {split_sk:.2f} merge_sk: {merge_sk:.2f}')

    # (are, precision, recall) = adapted_rand(seg=pred, gt=gt, all_stats=True)
    # print(f'{stackname} - are: {are:.2f} precision: {precision:.2f} recall: {recall:.2f}')
    are_sk, precision_sk, recall_sk = adapted_rand_error(image_true=gt, image_test=pred, ignore_labels=[0, ])
    print(f'{stackname} - are_sk: {are_sk:.2f} precision_sk: {precision_sk:.2f} recall_sk: {recall_sk:.2f}')

    # commented because it take a lot of ram, which is a problem on very large stacks.
    # homogeneity: each cluster contains only members of a single class.
    # completeness: all members of a given class are assigned to the same cluster.
    # v_measure: harmonic mean of homogeneity and completeness
    # homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(labels_true=gt.ravel(), labels_pred=pred.ravel())
    # print(f'{stackname} - homogeneity: {homogeneity:.2f} completeness: {completeness:.2f} v_measure: {v_measure:.2f}')

    matching_results = matching(y_true=gt, y_pred=pred, thresh=0.5, criterion='iou', report_matches=True)
    accuracy_stardist, f1_stardist, fn_stardist, fp_stardist, tp_stardist = matching_results.accuracy, matching_results.f1, matching_results.fn, matching_results.fp, matching_results.tp
    mean_matched_score_stardist = matching_results.mean_matched_score
    mean_true_score_stardist = matching_results.mean_true_score
    panoptic_quality_stardist = matching_results.panoptic_quality
    n_pred_stardist = matching_results.n_pred
    n_true_stardist = matching_results.n_true
    precision_stardist = matching_results.precision
    recall_stardist = matching_results.recall
    print(f'{stackname} - Stats are with 0.5 threshold for IOU!!')
    print(
        f'{stackname} - accuracy: {accuracy_stardist:.2f} precision: {precision_stardist:.2f} recall: {recall_stardist:.2f} f1: {f1_stardist:.2f}')
    print(
        f'{stackname} - mean_matched_score: {mean_matched_score_stardist:.2f} mean_true_score: {mean_true_score_stardist:.2f} panoptic_quality: {panoptic_quality_stardist:.2f}')
    print(
        f'{stackname} - n_pred: {n_pred_stardist} n_true: {n_true_stardist} fn: {fn_stardist} fp: {fp_stardist} tp: {tp_stardist}')

    df_single = pd.DataFrame({
        'accuracy_stardist': [accuracy_stardist],
        'f1_stardist': [f1_stardist],
        'precision_stardist': [precision_stardist],
        'recall_stardist': [recall_stardist],
        'mean_matched_score_stardist': [mean_matched_score_stardist],
        'mean_true_score_stardist': [mean_true_score_stardist],
        'panoptic_quality_stardist': [panoptic_quality_stardist],
        'fn_stardist': [fn_stardist],
        'fp_stardist': [fp_stardist],
        'tp_stardist': [tp_stardist],
        'n_pred_stardist': [n_pred_stardist],
        'n_true_stardist': [n_true_stardist],
        # 'split': [split],
        'split_sk': [split_sk],
#         'merge': [merge],
        'merge_sk': [merge_sk],
#         'are': [are],
        'are_sk': [are_sk],
#         'precision': [precision],
        'precision_sk': [precision_sk],
#         'recall': [recall],
        'recall_sk': [recall_sk],
        # 'homogeneity': [homogeneity],
        # 'completeness': [completeness],
        # 'v_measure': [v_measure],
        'stackname': [stackname],
        "foldername_pred": os.path.basename(pred_path),
        "foldername_gt": os.path.basename(gt_path)
    }, index=[stackname])

    return df_single

# def evaluate_segmentation(path_to_gt, path_to_pred, file_ext='nrrd'):
def evaluate_segmentation(path_gt: str, path_pred: str,
                          file_ext_gt: str = 'nrrd',
                          file_ext_pred: str = 'nrrd',
                          crop_to_fit: bool = False,
                          path_xlsx_out: str = 'results/output.xlsx'):
    '''

    :param file_ext_pred:
    :param path_xlsx_out:
    :param path_gt: folder containing the ground truth images
    :param path_pred: folder containing the predicted images
    :param file_ext_gt: file extension of the images
    :param crop_to_fit: crop the ground truth to fit the prediction, used to fix an inference issue on fabio's side
    :return:
    '''

    df = pd.DataFrame()

    foldername_pred = os.path.basename(path_pred)
    foldername_gt = os.path.basename(path_gt)


    pred_files = glob.glob(f'{path_pred}/*.{file_ext_pred}')
    for pred_path in pred_files:
        stackname = os.path.basename(pred_path).split('.')[0]
        stackname = stackname.replace("cell_segm_", "")
        if '_fx' in stackname:
            stackname = stackname.split('_fx')[0]

        # check if exists
        gt_path = f'{path_gt}/{stackname}.{file_ext_gt}'
        if not os.path.exists(gt_path):
            print(f'GT file not found: {gt_path}')
            continue

        # load images
        pred = itk.GetArrayFromImage(itk.imread(pred_path))

        # check if fits in uint32
        max_val = pred.max()
        min_val = pred.min()
        if max_val > 2 ** 32 - 1:
            print(f'Image exceeds uint32: {max_val}')
            continue
        if min_val < 0:
            print(f'Image has negative values: {min_val}')
            continue
        pred = pred.astype(np.uint32)
        gt = itk.GetArrayFromImage(itk.imread(gt_path))

        # inference issue on fabio's side, his suggestion was to crop the gt to fit the pred
        pred, gt = match_pred_and_gt_shape(pred, gt, crop_to_fit=crop_to_fit)
        assert pred.shape == gt.shape, f'Shape mismatch: {pred.shape} // {gt.shape}'

        # (split, merge) = voi(pred, gt)
        # print(f'{stackname} - split: {split:.2f} merge: {merge:.2f}')
        split_sk, merge_sk = variation_of_information(gt, pred, ignore_labels=[0, ])
        print(f'{stackname} - split_sk: {split_sk:.2f} merge_sk: {merge_sk:.2f}')

        # (are, precision, recall) = adapted_rand(seg=pred, gt=gt, all_stats=True)
        # print(f'{stackname} - are: {are:.2f} precision: {precision:.2f} recall: {recall:.2f}')
        are_sk, precision_sk, recall_sk = adapted_rand_error(image_true=gt, image_test=pred, ignore_labels=[0, ])
        print(f'{stackname} - are_sk: {are_sk:.2f} precision_sk: {precision_sk:.2f} recall_sk: {recall_sk:.2f}')

        # commented because it take a lot of ram, which is a problem on very large stacks.
        # homogeneity: each cluster contains only members of a single class.
        # completeness: all members of a given class are assigned to the same cluster.
        # v_measure: harmonic mean of homogeneity and completeness
        # homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(labels_true=gt.ravel(), labels_pred=pred.ravel())
        # print(f'{stackname} - homogeneity: {homogeneity:.2f} completeness: {completeness:.2f} v_measure: {v_measure:.2f}')

        matching_results = matching(y_true=gt, y_pred=pred, thresh=0.5, criterion='iou', report_matches=True)
        accuracy_stardist, f1_stardist, fn_stardist, fp_stardist, tp_stardist = matching_results.accuracy, matching_results.f1, matching_results.fn, matching_results.fp, matching_results.tp
        mean_matched_score_stardist = matching_results.mean_matched_score
        mean_true_score_stardist = matching_results.mean_true_score
        panoptic_quality_stardist = matching_results.panoptic_quality
        n_pred_stardist = matching_results.n_pred
        n_true_stardist = matching_results.n_true
        precision_stardist = matching_results.precision
        recall_stardist = matching_results.recall
        print(f'{stackname} - Stats are with 0.5 threshold for IOU!!')
        print(
            f'{stackname} - accuracy: {accuracy_stardist:.2f} precision: {precision_stardist:.2f} recall: {recall_stardist:.2f} f1: {f1_stardist:.2f}')
        print(
            f'{stackname} - mean_matched_score: {mean_matched_score_stardist:.2f} mean_true_score: {mean_true_score_stardist:.2f} panoptic_quality: {panoptic_quality_stardist:.2f}')
        print(
            f'{stackname} - n_pred: {n_pred_stardist} n_true: {n_true_stardist} fn: {fn_stardist} fp: {fp_stardist} tp: {tp_stardist}')

        df_single = pd.DataFrame({
            'accuracy_stardist': [accuracy_stardist],
            'f1_stardist': [f1_stardist],
            'precision_stardist': [precision_stardist],
            'recall_stardist': [recall_stardist],
            'mean_matched_score_stardist': [mean_matched_score_stardist],
            'mean_true_score_stardist': [mean_true_score_stardist],
            'panoptic_quality_stardist': [panoptic_quality_stardist],
            'fn_stardist': [fn_stardist],
            'fp_stardist': [fp_stardist],
            'tp_stardist': [tp_stardist],
            'n_pred_stardist': [n_pred_stardist],
            'n_true_stardist': [n_true_stardist],
            # 'split': [split],
            'split_sk': [split_sk],
#             'merge': [merge],
            'merge_sk': [merge_sk],
#             'are': [are],
            'are_sk': [are_sk],
#             'precision': [precision],
            'precision_sk': [precision_sk],
#             'recall': [recall],
            'recall_sk': [recall_sk],
            # 'homogeneity': [homogeneity],
            # 'completeness': [completeness],
            # 'v_measure': [v_measure],
            'stackname': [stackname],
            "foldername_pred": [foldername_pred],
            "foldername_gt": [foldername_gt]
        }, index=[stackname])

        df = pd.concat([df, df_single])

    dirname_output = os.path.dirname(path_xlsx_out)
    if not os.path.exists(dirname_output):
        os.makedirs(dirname_output)

    # if file exists, copy it to a backup file with a timestamp
    backup_folder = os.path.join(dirname_output, 'backup')
    if not os.path.exists(backup_folder):
        os.makedirs(backup_folder)
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    old_filename = os.path.splitext(os.path.basename(path_xlsx_out))[0]
    backup_file = os.path.join(backup_folder, f'{old_filename}_{timestamp}.xlsx')
    if os.path.exists(path_xlsx_out):
        shutil.copy(path_xlsx_out, backup_file)
        print(f'Backup of {path_xlsx_out} saved to {backup_file}')


    with pd.ExcelWriter(path_xlsx_out, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Sheet1')
        worksheet = writer.sheets['Sheet1']
        worksheet.autofit()
