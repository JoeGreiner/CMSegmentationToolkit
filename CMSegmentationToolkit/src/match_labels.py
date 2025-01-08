import os
import numpy as np
import glob
import itk
import pandas as pd
from stardist.matching import matching
import logging

df = pd.DataFrame()


def apply_label_mapping(seg: np.ndarray, mapping: dict):
    labels_in_pred = np.unique(seg)

    if seg.dtype != np.uint16:
        assert max(labels_in_pred) < 2 ** 16, f'Image exceeds uint16: {max(labels_in_pred)}'
        assert min(labels_in_pred) >= 0, f'Image has negative values: {min(labels_in_pred)}'
        seg = seg.astype(np.uint16)

    seg_itk = itk.GetImageFromArray(seg)

    change_label_filter = itk.ChangeLabelImageFilter[itk.Image[itk.US, 3], itk.Image[itk.US, 3]].New()
    change_label_filter.SetInput(seg_itk)
    for label_change_from, label_change_to in mapping.items():
        change_label_filter.SetChange(int(label_change_from), int(label_change_to))

    return itk.GetArrayFromImage(change_label_filter.GetOutput())

def match_prediction_to_gt(pred, gt):
    # iou matching
    matching_results = matching(y_true=gt, y_pred=pred, thresh=0.5, criterion='iou', report_matches=True)

    # get matched pairs

    matched_mapping = {}
    visited_labels = set()
    for gt_label, pred_label in matching_results.matched_pairs:
        matched_mapping[pred_label] = gt_label
        visited_labels.add(pred_label)

    # apply mapping
    pred_matched = apply_label_mapping(pred, matched_mapping)
    return pred_matched


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    file_ext = 'nrrd'
    path_to_gt = r'D:\data\confocal\wga\species_comparison\autoseg_paper\fabio\gt'
    path_to_pred = r'D:\data\confocal\wga\species_comparison\autoseg_paper\fabio\pred'

    shape_match_folder = r'D:\data\confocal\wga\species_comparison\autoseg_paper\fabio\shape_match'

    vis_export_folder = r'D:\data\confocal\wga\species_comparison\autoseg_paper\fabio\vis'
    fg_bg_error_folder = r'D:\data\confocal\wga\species_comparison\autoseg_paper\fabio\fg_bg_error'
    true_matches_folder = r'D:\data\confocal\wga\species_comparison\autoseg_paper\fabio\true_matches'
    pred_only_folder = r'D:\data\confocal\wga\species_comparison\autoseg_paper\fabio\pred_only'
    gt_only_folder = r'D:\data\confocal\wga\species_comparison\autoseg_paper\fabio\gt_only'

    if not os.path.exists(shape_match_folder):
        os.makedirs(shape_match_folder)

    if not os.path.exists(vis_export_folder):
        os.makedirs(vis_export_folder)

    if not os.path.exists(vis_export_folder):
        os.makedirs(vis_export_folder)

    if not os.path.exists(fg_bg_error_folder):
        os.makedirs(fg_bg_error_folder)

    pred_files = glob.glob(f'{path_to_pred}/*.{file_ext}')
    for pred_path in pred_files:
        stackname = os.path.basename(pred_path).split('.')[0]
        stackname = stackname.replace("cell_segm_", "")

        # check if exists
        gt_path = f'{path_to_gt}/{stackname}.{file_ext}'
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
        # check if swap would make it match
        if pred.shape != gt.shape:
            if pred.shape[0] == gt.shape[2] and pred.shape[2] == gt.shape[0]:
                print(f'Swapping axes to match: {pred.shape} // {gt.shape}')
                gt = np.swapaxes(gt, 0, 2).copy()

        # check shapes
        if pred.shape != gt.shape:
            print(f'Shape mismatch: {pred.shape} // {gt.shape}')
            continue
        else:
            print(f'Shape match: {pred.shape} // {gt.shape}')

        itk.imwrite(itk.GetImageFromArray(gt), f'{shape_match_folder}/{stackname}_gt.nrrd')
        itk.imwrite(itk.GetImageFromArray(pred), f'{shape_match_folder}/{stackname}_pred.nrrd')

        logging.info(f'Calculating matches for {stackname}')
        matching_results = matching(y_true=gt, y_pred=pred, thresh=0.5, criterion='iou', report_matches=True)
        logging.info(f'Finished calculating matches for {stackname}')
        # matching_results.matched_pairs # contains (gt, pred) labels
        accuracy_stardist, f1_stardist, fn_stardist, fp_stardist, tp_stardist = matching_results.accuracy, matching_results.f1, matching_results.fn, matching_results.fp, matching_results.tp
        mean_matched_score_stardist = matching_results.mean_matched_score
        mean_true_score_stardist = matching_results.mean_true_score
        panoptic_quality_stardist = matching_results.panoptic_quality
        n_pred_stardist = matching_results.n_pred
        n_true_stardist = matching_results.n_true
        precision_stardist = matching_results.precision
        recall_stardist = matching_results.recall
        print(f'{stackname} - Stats are with threshold for IOU!!')
        print(f'{stackname} - accuracy: {accuracy_stardist:.2f} precision: {precision_stardist:.2f} recall: {recall_stardist:.2f} f1: {f1_stardist:.2f}')
        print(f'{stackname} - mean_matched_score: {mean_matched_score_stardist:.2f} mean_true_score: {mean_true_score_stardist:.2f} panoptic_quality: {panoptic_quality_stardist:.2f}')
        print(f'{stackname} - n_pred: {n_pred_stardist} n_true: {n_true_stardist} fn: {fn_stardist} fp: {fp_stardist} tp: {tp_stardist}')



        matched_mapping = {}
        unique_labels_pred = np.unique(pred)
        visited_labels = set()
        for gt_label, pred_label in matching_results.matched_pairs:
            matched_mapping[pred_label] = gt_label
            visited_labels.add(pred_label)
        #
        # non_visited_labels = set(unique_labels_pred) - visited_labels

        # for ix, label in enumerate(non_visited_labels):
        #     matched_mapping[label] = max(unique_labels_pred) + ix + 1

        logging.info(f'Applying label mapping to seg')
        pred_matched = apply_label_mapping(pred, matched_mapping)
        logging.info(f'Finished applying label mapping to seg')



        #
        # logging.info(f'Calculating matches for {stackname}')
        # matches = find_matches(ref=LabeledSegmentation(gt), pred=LabeledSegmentation(pred))
        # logging.info(f'Finished calculating matches for {stackname}')
        #
        # in_ref_only = matches['in_ref_only']
        # in_pred_only = matches['in_pred_only']
        # true_matches = matches['true_matches']
        # true_matches_IoU = matches['true_matches_IoU']
        #
        # labels_in_pred = np.unique(pred)
        # labels_in_gt = np.unique(gt)
        #
        # matched_pred = np.zeros_like(pred)
        #
        # # create zero mapping
        # matched_mapping = {}
        # max_label_gt = max(labels_in_gt)
        # for ix, label in enumerate(in_pred_only):
        #     matched_mapping[label] = max_label_gt + ix + 1
        #
        # for ref_label, pred_label in true_matches:
        #     matched_mapping[pred_label] = ref_label
        #
        # logging.info(f'Applying label mapping to seg')
        # pred_matched = apply_label_mapping(pred, matched_mapping)
        # logging.info(f'Finished applying label mapping to seg')

        confusion_image = np.zeros_like(pred_matched)
        confusion_image[(gt == pred_matched) & (gt != 0)] = 1  # true positive
        confusion_image[(gt == 0) & (pred_matched == 0)] = 2  # true negative
        confusion_image[(pred_matched > 0) & (pred_matched != gt)] = 3  # false positive
        confusion_image[(gt != 0) & (pred_matched == 0)] = 4  # false negative
        TP = np.sum(confusion_image == 1)
        FP = np.sum(confusion_image == 3)
        FN = np.sum(confusion_image == 4)
        TN = np.sum(confusion_image == 2)
        accuracy = (TP + TN) / (TP + FP + FN + TN)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * (precision * recall) / (precision + recall)

        print(f'{stackname} - Stats are without threshold for IOU!!')
        print(f'{stackname} - TP: {TP} FP: {FP} FN: {FN} TN: {TN}')
        print(f'{stackname} - accuracy: {accuracy:.2f} precision: {precision:.2f} recall: {recall:.2f} f1: {f1:.2f}')


        # todo: write all errors, also e.g. where there were merge errors!
        # ie, take true matches in pred, get gt with only matching labels, and then find non-matching labels

        # write
        itk.imwrite(itk.GetImageFromArray(pred_matched), f'{fg_bg_error_folder}/{stackname}_matched.nrrd')
        itk.imwrite(itk.GetImageFromArray(confusion_image), f'{fg_bg_error_folder}/{stackname}_confusion.nrrd')
