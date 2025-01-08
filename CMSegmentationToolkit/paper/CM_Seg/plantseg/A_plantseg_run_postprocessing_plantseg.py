import glob
import os
import itk
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm

from CMSegmentationToolkit.paper.CM_Seg.dtwsmc.A1_run_tuning import evaluate_segmentation
import SimpleITK as sitk

def load_npz(input_file, channel=0):
    logging.debug(f'loading {input_file}')
    img = np.load(input_file, mmap_mode='c')['probabilities'][channel,]  # zyx
    logging.debug(f'loading {input_file} done.')
    logging.debug(f'img.shape: {img.shape} (should be xyz)')
    return img

def load_h5(input_file, channel=0):
    logging.debug(f'loading {input_file}')
    import h5py
    with h5py.File(input_file, 'r') as f:
        img = np.array(f['segmentation'])  # zyx
    logging.debug(f'loading {input_file} done.')
    logging.debug(f'img.shape: {img.shape} (should be xyz)')
    return img

def mask_segments_with_prediction(input_file_segments, input_file_probabilties, output_file, compress=True, threshold=0.5, background_label=0,
                                  cell_mask_channel=1, export_probabilities=False, swap_axes=True):
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    logging.debug(f'masking segments with prediction on {input_file_segments}')
    logging.debug(f'writing to {output_file}')
    logging.debug(f'compress: {compress}')
    logging.debug(f'threshold: {threshold}')
    logging.debug(f'background_label: {background_label}')

    if not isinstance(input_file_segments, np.ndarray):
        assert os.path.exists(input_file_segments), f'input_file_segments {input_file_segments} does not exist'
    if not isinstance(input_file_probabilties, np.ndarray):
        assert os.path.exists(input_file_probabilties), f'input_file_probabilties {input_file_probabilties} does not exist'

    if not os.path.exists(os.path.dirname(output_file)):
        logging.debug(f'creating {os.path.dirname(output_file)}')
        if os.path.dirname(output_file) != '':
            os.makedirs(os.path.dirname(output_file))

    if isinstance(input_file_segments, np.ndarray):
        img = input_file_segments
    elif input_file_segments.endswith('.npz'):
        img = load_npz(input_file_segments)
    elif input_file_segments.endswith('.h5'):
        img = load_h5(input_file_segments)
    else:
        img = sitk.ReadImage(input_file_segments)
        img = sitk.GetArrayFromImage(img)
    logging.debug(f'img.shape: {img.shape} (should be zyx)')

    if isinstance(input_file_probabilties, np.ndarray):
        img_prob = input_file_probabilties
    elif input_file_probabilties.endswith('.npz'):
        img_prob = load_npz(input_file_probabilties, channel=cell_mask_channel)
    elif input_file_probabilties.endswith('.h5'):
        img_prob = load_h5(input_file_probabilties, channel=cell_mask_channel)
    else:
        img_prob = sitk.ReadImage(input_file_probabilties)
        img_prob = sitk.GetArrayFromImage(img_prob)
    logging.debug(f'img_prob.shape: {img_prob.shape} (should be zyx)')

    if swap_axes:
        logging.debug(f'swapping axes, should now be zyx after write')
        img = np.swapaxes(img, 0, 2).copy()
        img_prob = np.swapaxes(img_prob, 0, 2).copy()
        logging.debug(f'img.shape: {img.shape} (should be zyx)')
        logging.debug(f'img_prob.shape: {img_prob.shape} (should be zyx)')

    if export_probabilities:
        logging.debug(f'writing probabilities to {output_file.replace(".nrrd", "_probabilities.nrrd")}')

        sitk.WriteImage(sitk.GetImageFromArray(img_prob), output_file.replace(".nrrd", "_probabilities.nrrd"), useCompression=compress)

    # from matplotlib import pyplot as plt
    # plt.figure()
    # plt.imshow(img_prob[img_prob.shape[0]//2])
    # plt.show()

    assert img.shape == img_prob.shape, f'img.shape: {img.shape}, img_prob.shape: {img_prob.shape}'

    # use LabelIntensityStatisticsImageFilter  from itk for fast implementation
    label_intensity_stats = sitk.LabelIntensityStatisticsImageFilter()
    label_intensity_stats.Execute(sitk.GetImageFromArray(img), sitk.GetImageFromArray(img_prob))

    # get labels
    labels = label_intensity_stats.GetLabels()
    labels_to_change_to_bg = []
    for label in labels:
        if label == background_label:
            continue
        mean = label_intensity_stats.GetMean(int(label))
        if mean < threshold:
            # dont do masks directly, use changelabelfilter later for faster speed
            # img[img == label] = background_label
            labels_to_change_to_bg.append(label)


    if img.dtype != np.uint16:
        assert max(labels) < 2 ** 16, f'Image exceeds uint16: {max(labels)}'
        assert min(labels) >= 0, f'Image has negative values: {min(labels)}'
        img = img.astype(np.uint16)

    change_label_image_filter = sitk.ChangeLabelImageFilter()
    change_map = {}
    for label in labels_to_change_to_bg:
        change_map[label] = background_label
    logging.debug(f'changing labels to background f{background_label}: {labels_to_change_to_bg}')
    change_label_image_filter.SetChangeMap(change_map)

    logging.debug(f'changing labels')
    img = change_label_image_filter.Execute(sitk.GetImageFromArray(img))
    logging.debug(f'changing labels done')

    logging.debug(f'writing {output_file}')
    sitk.WriteImage(img, output_file, useCompression=compress)
    logging.debug(f'writing {output_file} done.')


def run_masking_for_validation_with_gt(skip_if_output_file_exists=True):
    multicuts = ['05', '06', '07', '08', '09']
    path_to_labels = '/mnt/work/data/CM_Seg_Paper_24/data_v2/'
    for multicut_number in multicuts:
        path_to_results = f'/mnt/work/data/CM_Seg_Paper_24/results/plantseg/2024_10_02_Validation/MultiCut_{multicut_number}/'
        all_h5 = glob.glob(os.path.join(path_to_results, '*.h5'))

        for h5 in all_h5:
            stackname = os.path.basename(h5).replace('_predictions_multicut.h5', '')
            matching_labels = glob.glob(os.path.join(path_to_labels, '*', 'train', 'labels', f'{stackname}.nrrd'))

            assert len(matching_labels) == 1, f'Found {len(matching_labels)} matching labels for {stackname}'

            folder_input_prediction = os.path.dirname(h5)
            folder_output = os.path.join(folder_input_prediction, 'prediction_masked')
            path_output = os.path.join(folder_output, f'{stackname}.nrrd')
            if skip_if_output_file_exists and os.path.exists(path_output):
                logging.info(f'Skipping {stackname} because output file exists')
                continue

            label_img = itk.GetArrayFromImage(itk.imread(matching_labels[0]))
            label_img = (label_img > 0).astype(np.uint8)

            mask_segments_with_prediction(input_file_segments=h5, input_file_probabilties=label_img,
                                          output_file=path_output, compress=True, threshold=0.5, background_label=0,
                                          export_probabilities=False, swap_axes=False)

def run_evaluation_for_validation_with_gt():
    df = pd.DataFrame()

    multicuts = ['05', '06', '07', '08', '09']
    path_to_labels = '/mnt/work/data/CM_Seg_Paper_24/data_v2/'
    for multicut_number in multicuts:
        path_to_results = f'/mnt/work/data/CM_Seg_Paper_24/results/plantseg/2024_10_02_Validation/MultiCut_{multicut_number}/prediction_masked/'
        all_masked_predictions = glob.glob(os.path.join(path_to_results, '*.nrrd'))

        for prediction in tqdm(all_masked_predictions):
            stackname = os.path.basename(prediction).replace('.nrrd', '')
            matching_labels = glob.glob(os.path.join(path_to_labels, '*', 'train', 'labels', f'{stackname}.nrrd'))

            # get whatever foldername is between path_to_labels and train
            foldername = os.path.dirname(matching_labels[0]).split('/')[-3]
            assert len(matching_labels) == 1, f'Found {len(matching_labels)} matching labels for {stackname}'

            evaluation = evaluate_segmentation(prediction, matching_labels[0])
            evaluation['stackname'] = stackname
            evaluation['multicut'] = multicut_number
            evaluation['dataset'] = foldername

            df = pd.concat([df, evaluation])

    df.to_excel('multicut_tuning_validation_results.xlsx')


    min_mean_adaptive_rand_error = df.groupby('multicut').mean(numeric_only=True)
    min_mean_adaptive_rand_error.to_excel('multicut_tuning_validation_results.xlsx')
    print(min_mean_adaptive_rand_error)

    datasets = df['dataset'].unique()
    for dataset in datasets:
        min_mean_adaptive_rand_error = df[df['dataset'] == dataset].groupby('multicut').mean(numeric_only=True)
        min_mean_adaptive_rand_error.to_excel(f'multicut_tuning_validation_results_{dataset}.xlsx')
        print(min_mean_adaptive_rand_error)

def run_masking_for_test_with_prediction():
    path_to_results = '/mnt/work/data/CM_Seg_Paper_24/results/plantseg/2024_10_06_test/MultiCut_08/'
    all_h5 = glob.glob(os.path.join(path_to_results, '*.h5'))

    path_to_probabilities = '/mnt/work/data/CM_Seg_Paper_24/results/nnunet_2024_10_02_task_822/test/predictions/'

    for h5 in all_h5:
        stackname = os.path.basename(h5).replace('_predictions_multicut.h5', '')
        folder_input_prediction = os.path.dirname(h5)
        folder_output = os.path.join(folder_input_prediction, 'prediction_masked')
        path_output = os.path.join(folder_output, f'{stackname}.nrrd')

        matching_npz = os.path.join(path_to_probabilities, f'{stackname}.npz')
        if not os.path.exists(matching_npz):
            logging.info(f'Skipping {stackname} because matching npz does not exist')
            continue

        mask_segments_with_prediction(input_file_segments=h5, input_file_probabilties=matching_npz,
                                      output_file=path_output, compress=True, threshold=0.5, background_label=0,
                                      export_probabilities=False, swap_axes=False)




if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # for validation tuning, use the ground truth label masks
    run_masking_for_validation_with_gt()
    run_masking_for_test_with_prediction()

    run_evaluation_for_validation_with_gt()


