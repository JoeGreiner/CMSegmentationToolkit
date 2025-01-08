import glob
import os
import itk
import numpy as np
import pandas as pd
from skimage.metrics import variation_of_information, adapted_rand_error
from stardist.matching import matching
from tqdm import tqdm
from CMSegmentationToolkit.src.dtws_mc import run_freiburg_mc

def evaluate_segmentation(path_gt: str,
                          path_pred: str,
                          file_ext_pred: str = 'nrrd'):

    stackname = os.path.basename(path_pred).replace(f'.{file_ext_pred}', '')
    # pmap zer ogo
    stackname = stackname.replace('_pmap_zero', '')

    foldername_pred = os.path.basename(path_pred)
    foldername_gt = os.path.basename(path_gt)

    pred = itk.GetArrayFromImage(itk.imread(path_pred))
    pred = pred.astype(np.uint32)
    gt = itk.GetArrayFromImage(itk.imread(path_gt))

    # inference issue on fabio's side, his suggestion was to crop the gt to fit the pred
    assert pred.shape == gt.shape, f'Shape mismatch: {pred.shape} // {gt.shape}'

    split_sk, merge_sk = variation_of_information(gt, pred, ignore_labels=[0, ])
    print(f'{stackname} - split_sk: {split_sk:.2f} merge_sk: {merge_sk:.2f}')

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
        'split_sk': [split_sk],
        'merge_sk': [merge_sk],
        'are_sk': [are_sk],
        'precision_sk': [precision_sk],
        'recall_sk': [recall_sk],
        # 'homogeneity': [homogeneity],
        # 'completeness': [completeness],
        # 'v_measure': [v_measure],
        'stackname': [stackname],
        "foldername_pred": [foldername_pred],
        "foldername_gt": [foldername_gt]
    }, index=[stackname])

    return df_single

def get_mean_merge_split(gt, pred, file_ext='nrrd'):
    split_sk, merge_sk = variation_of_information(gt, pred, ignore_labels=[0, ])
    return {'split_sk': split_sk, 'merge_sk': merge_sk, 'mean_merge_split': (split_sk + merge_sk) / 2}



if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')

    val_fold_0 = ["170615_4_s7", "211123_1", "211123_5", "211216_2", "211216_5",
                  "exp_02_04_21_mouse_delta_1_b", "exp_02_04_21_mouse_delta_1_c",
                  "exp_04_11_20_pig_alpha_1", "exp_06_01_22_rat_delta_1_a", "exp_10_10_22_horse_521_6_dapi",
                  "exp_11_08_21_pig_gamma_2_a", 	"exp_11_08_21_pig_gamma_3_b"]

    path_to_df = 'multicut_tuning_vali_fold_0.xlsx'
    if os.path.exists(path_to_df):
        df = pd.read_excel(path_to_df)
    else:
        df = pd.DataFrame()


    for img_stackname in val_fold_0:
        path_to_input_file = glob.glob(f'/mnt/work/data/CM_Seg_Paper_24/data_v2/*/train/imgs_norm/{img_stackname}.nrrd', recursive=True)[0]
        path_to_labels = glob.glob(f'/mnt/work/data/CM_Seg_Paper_24/data_v2/*/train/labels/{img_stackname}.nrrd', recursive=True)[0]
        assert os.path.exists(path_to_input_file), f'File not found: {path_to_input_file}'
        assert os.path.exists(path_to_labels), f'File not found: {path_to_labels}'
        path_to_output_folder = 'tmp/'
        pickle_path = 'multicut_values.pkl'
        run_freiburg_mc(path_to_input_file, path_to_output_folder, betas=[0.05, 0.075, 0.1, 0.2], betas2=[], folds=[0],
                        relabel_seg=False)

        path_to_mcs = f'/home/greinerj/PycharmProjects/CM_Seg/tune_multicut/tmp/prediction_combined/{img_stackname}_MultiCut/'
        all_mcs = glob.glob(os.path.join(path_to_mcs, '*_pmap_zero.nrrd'))
        gt = itk.GetArrayFromImage(itk.imread(path_to_labels))
        for mc in tqdm(all_mcs, desc='Evaluating single multicuts'):
            mc_stackname = os.path.basename(mc)
            stackname_index = mc_stackname.replace('_pmap_zero.nrrd', '')
            mc_value = mc_stackname.replace(f'{img_stackname}_mc_', '').replace('_pmap_zero.nrrd', '')

            # check if is in df using full path
            if df.shape[0] > 0:
                if mc in df['full_path_mc'].values:
                    continue

            pred = itk.GetArrayFromImage(itk.imread(mc))
            # mean_merge_split = get_mean_merge_split(gt, pred)
            evaluation = evaluate_segmentation(path_gt=path_to_labels, path_pred=mc)
            # '170105_1_mc_0.01_pmap_zero.nrrd', -> 0.01
            # list.append({'mc_value': mc_value, 'mean_merge_split': mean_merge_split['mean_merge_split'],
            #              'split_sk': mean_merge_split['split_sk'], 'merge_sk': mean_merge_split['merge_sk'],
            #              'mc': mc_stackname, 'stackname': img_stackname})
            evaluation['mc_value'] = mc_value
            evaluation['mc'] = mc_stackname
            evaluation['full_path_mc'] = mc
            evaluation['stackname'] = img_stackname
            df = pd.concat([df, evaluation])


        print('\n+-----------------------------------+\n')
        # e.g. '/home/greinerj/PycharmProjects/CM_Seg/dtwsmc/tmp/prediction_combined/170105_1_MultiCut/170105_1_mc_0.5_MultiCut2/170105_1_mc_0.9_pmap_zero.nrrd'
        # evaluate the double multicuts

        all_double_multicuts = glob.glob(os.path.join(path_to_mcs, '*_MultiCut2', '*_pmap_zero.nrrd'), recursive=True)
        for double_mc in tqdm(all_double_multicuts, desc='Evaluating double multicuts'):

            # check if is in df using full path
            if df.shape[0] > 0:
                if double_mc in df['full_path_mc'].values:
                    continue

            mc_stackname2 = os.path.basename(double_mc)
            mc_foldername = os.path.basename(os.path.dirname(double_mc))
            # '170105_1_mc_0.01_pmap_zero.nrrd', -> 0.01
            mc2_value = mc_stackname2.replace(f'{img_stackname}_mc_', '').replace('_pmap_zero.nrrd', '')
            mc1_value = mc_foldername.replace(f'{img_stackname}_mc_', '').replace('_MultiCut2', '')
            mc_stackname = f'{mc1_value}_{mc2_value}_pmap_zero.nrrd'

            pred = itk.GetArrayFromImage(itk.imread(double_mc))
            # mean_merge_split = get_mean_merge_split(gt, pred)
            #
            # list.append({'mc_value': mc1_value, 'mc2_value': mc2_value,
            #              'mean_merge_split': mean_merge_split['mean_merge_split'],
            #              'split_sk': mean_merge_split['split_sk'], 'merge_sk': mean_merge_split['merge_sk'], 'mc': mc_stackname, 'stackname': img_stackname})
            evaluation = evaluate_segmentation(path_gt=path_to_labels, path_pred=double_mc)
            evaluation['mc_value'] = mc1_value
            evaluation['mc2_value'] = mc2_value
            evaluation['mc'] = mc_stackname
            evaluation['full_path_mc'] = double_mc
            evaluation['stackname'] = img_stackname
            df = pd.concat([df, evaluation])

    # create xlsx
    df.to_excel('multicut_tuning_vali_fold_0.xlsx')



