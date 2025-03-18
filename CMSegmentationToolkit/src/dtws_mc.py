import glob
import logging
import os
import subprocess
from os.path import dirname, join
import numpy as np
import itk
from elf.segmentation import compute_rag, compute_boundary_features, compute_boundary_mean_and_length, \
    project_node_labels_to_pixels
from elf.segmentation.multicut import transform_probabilities_to_costs, multicut_kernighan_lin
from elf.segmentation.watershed import blockwise_two_pass_watershed
from skimage.segmentation import relabel_sequential
from tqdm import tqdm

def set_threshold_img_to_bg(segmentation, image, threshold=0.5, bg_value='zero'):
    altered_seg = segmentation.copy()
    if bg_value == 'max':
        altered_seg[image > threshold] = np.max(altered_seg) + 1
    elif bg_value == 'zero':
        altered_seg[altered_seg == 0] = np.max(altered_seg) + 1
        altered_seg[image > threshold] = 0
    return altered_seg

def convert_labels_to_smaller_dtype(in_label_array):
    # cast to smaller dtype depending on max value
    if in_label_array.max() < 256:
        in_label_array = in_label_array.astype(np.uint8)
    elif in_label_array.max() < 65536:
        in_label_array = in_label_array.astype(np.uint16)
    else:
        in_label_array = in_label_array.astype(np.uint32)
    return in_label_array

def run_multicut(path_to_prediction, path_to_watershed, output_folder, betas=[0.25, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
                 n_threads=None, invert_boundary_input=False,
                 export_non_pmap_supervoxels=True, skip_if_multicut_exists=True, boundary_threshold=0.5,
                 swapaxes=False, relabel_seg=True):
    betas = betas.copy() # make a copy to not change the original list
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(f'path_to_prediction: {path_to_prediction}')
    logging.info(f'path_to_watershed: {path_to_watershed}')
    logging.info(f'output_folder: {output_folder}')
    logging.info(f'betas: {betas}')
    logging.info(f'n_threads: {n_threads}')
    logging.info(f'export_non_pmap_supervoxels: {export_non_pmap_supervoxels}')
    logging.info(f'skip_if_multicut_exists: {skip_if_multicut_exists}')
    logging.info(f'boundary_threshold: {boundary_threshold}')
    logging.info(f'swapaxes: {swapaxes}')

    current_file_ending = os.path.splitext(os.path.basename(path_to_prediction))[1]
    # if ending is .gz, check if it is .nrrd.gz or .nii.gz
    if current_file_ending == '.gz':
        if '.nrrd.gz' in path_to_prediction:
            current_file_ending = '.nrrd.gz'
        elif '.nii.gz' in path_to_prediction:
            current_file_ending = '.nii.gz'
    logging.info(f'file ending: {current_file_ending}')

    if skip_if_multicut_exists:
        betas_to_remove = []
        for beta in betas:
            output_path = os.path.join(output_folder,
                                       os.path.basename(path_to_prediction).replace(f'{current_file_ending}',
                                                                                    f'_mc_{beta}{current_file_ending}'))
            if os.path.exists(output_path):
                logging.info(f'Skipping {path_to_prediction} because {output_path} exists')
                betas_to_remove.append(beta)
        for beta in betas_to_remove:
            betas.remove(beta)


    if len(betas) == 0:
        logging.info(f'All betas already exist, skipping {path_to_prediction}')
        return

    logging.info(f'Running Multicut on {path_to_prediction}')
    logging.info(f'Output will be saved to {output_folder}')

    logging.info(f'Loading {path_to_prediction}')
    boundary_input_zyx = itk.GetArrayFromImage(itk.imread(path_to_prediction))
    if swapaxes:
        logging.info(f'Swapping axes of {path_to_prediction}')
        boundary_input_zyx = np.swapaxes(boundary_input_zyx, 0, 2)
    if boundary_input_zyx.max() > 1:
        logging.info(f'Normalizing {path_to_prediction}')
        boundary_input_zyx = (boundary_input_zyx / 255).astype(np.float32)
    else:
        logging.info(f'Not normalizing {path_to_prediction}')
        boundary_input_zyx = boundary_input_zyx.astype(np.float32)

    if invert_boundary_input:
        logging.info(f'Inverting {path_to_prediction}')
        boundary_input_zyx = 1 - boundary_input_zyx

    logging.info(f'Min: {boundary_input_zyx.min()}')
    logging.info(f'Max: {boundary_input_zyx.max()}')
    logging.info(f'Mean: {boundary_input_zyx.mean()}')
    logging.info(f'Median: {np.median(boundary_input_zyx)}')

    logging.info(f'Loading {path_to_watershed}')
    watershed = itk.GetArrayFromImage(itk.imread(path_to_watershed))

    if swapaxes:
        watershed = np.swapaxes(watershed, 0, 2)

    logging.info(f'watershed.shape: {watershed.shape}')
    logging.info(f'boundary_input_zyx.shape: {boundary_input_zyx.shape}')

    assert watershed.shape == boundary_input_zyx.shape, f'Watershed and boundary input have different shapes: {watershed.shape} vs {boundary_input_zyx.shape}'

    # import napari
    # viewer = napari.view_image(boundary_input_zyx)
    # viewer.add_labels(watershed)
    # napari.run()

    logging.info('computing RAG')
    rag = compute_rag(watershed, n_threads=n_threads)
    logging.info('computing boundary features')
    boundary_features = compute_boundary_features(rag, boundary_input_zyx, n_threads=n_threads)[:, 0]
    logging.info('computing edge sizes')
    edge_sizes = compute_boundary_mean_and_length(rag, boundary_input_zyx)[:, 1]

    for beta in tqdm(betas, desc='Calculating Multicuts'):
        # calculate multicut
        logging.info(f'Calculating Multicut with beta={beta}')
        costs = transform_probabilities_to_costs(boundary_features, edge_sizes=edge_sizes, beta=beta)

        # solve with kerninghan lin
        logging.info('Solving Multicut')
        node_labels = multicut_kernighan_lin(rag, costs)
        #
        # block_shape_mc = (64, 512, 512)
        # node_labels = blockwise_multicut(graph=rag, costs=costs, internal_solver="kernighan-lin",
        #                                     segmentation=watershed, block_shape=block_shape_mc, n_threads=n_threads)

        logging.info('Projecting node labels to pixels')
        seg = project_node_labels_to_pixels(rag, node_labels, n_threads)

        # import napari
        # viewer = napari.view_image(boundary_input_zyx)
        # viewer.add_labels(seg)
        # viewer.add_labels(watershed)
        # napari.run()

        if relabel_seg:
            logging.info('Relabeling')
            seg = relabel_sequential(seg)[0]


        output_path_seg = os.path.join(output_folder,
                                       os.path.basename(path_to_prediction).replace(f'{current_file_ending}', f'_mc_{beta}{current_file_ending}'))
        output_path_seg_pmap_zero = os.path.join(output_folder,
                                                 os.path.basename(path_to_prediction).replace(f'{current_file_ending}',
                                                                                              f'_mc_{beta}_pmap_zero{current_file_ending}'))
        seg = convert_labels_to_smaller_dtype(seg)
        if export_non_pmap_supervoxels:
            logging.info('Exporting non-pmap supervoxels')
            seg = itk.GetImageFromArray(seg)
            itk.imwrite(seg, output_path_seg, compression=True)
            seg = itk.GetArrayFromImage(seg)

        logging.info('Setting pmap zero and exporting')
        seg = set_threshold_img_to_bg(seg, boundary_input_zyx, threshold=boundary_threshold)
        seg = itk.GetImageFromArray(seg)
        itk.imwrite(seg, output_path_seg_pmap_zero, compression=True)
        logging.info('Done')


def calculate_two_pass_watershed(boundary_input_zyx, boundary_threshold, sigma_seeds=2.0,
                                 n_threads=None, verbose=True, min_size=100,
                                 # compactness=0
                                 ):

    block_shape = (64, 512, 512)
    halo = (12, 48, 48)

    assert boundary_input_zyx.ndim == 3, 'boundary input has to be 3d'
    assert boundary_input_zyx.max() <= 1, 'boundary input has to be in [0, 1]'
    assert boundary_input_zyx.min() >= 0, 'boundary input has to be in [0, 1]'
    assert boundary_threshold >= 0, 'boundary threshold has to be >= 0'
    assert boundary_threshold <= 1, 'boundary threshold has to be <= 1'
    assert sigma_seeds >= 0, 'sigma seeds has to be >= 0'
    assert min_size >= 0, 'min size has to be >= 0'
    # assert compactness >= 0, 'compactness has to be >= 0'

    if boundary_input_zyx.shape[0] < block_shape[0]:
        block_shape = (boundary_input_zyx.shape[0], block_shape[1], block_shape[2])
        halo = (0, halo[1], halo[2])
    if boundary_input_zyx.shape[1] < block_shape[1]:
        block_shape = (block_shape[0], boundary_input_zyx.shape[1], block_shape[2])
        halo = (halo[0], 0, halo[2])
    if boundary_input_zyx.shape[2] < block_shape[2]:
        block_shape = (block_shape[0], block_shape[1], boundary_input_zyx.shape[2])
        halo = (halo[0], halo[1], 0)

    try:
        watershed, _ = blockwise_two_pass_watershed(boundary_input_zyx, block_shape, halo,
                                                    verbose=verbose, threshold=boundary_threshold,
                                                    sigma_seeds=sigma_seeds,
                                                    n_threads=n_threads,
                                                    min_size=min_size,
                                                    # compactness=compactness
                                                    )
    except AssertionError:
        print('Blockwise watershed failed, trying again with smaller block shape')
        block_shape = (48, 256, 256)
        halo = (12, 48, 48)
        watershed, _ = blockwise_two_pass_watershed(boundary_input_zyx, block_shape, halo,
                                                    verbose=verbose, threshold=boundary_threshold,
                                                    sigma_seeds=sigma_seeds,
                                                    n_threads=n_threads,
                                                    min_size=min_size,
                                                    # compactness=compactness
                                                    )

    labels = np.unique(watershed)
    logging.info(f'Initial watershed: {len(labels)} segments')

    return watershed

def read_nnu_proba(input_file, channel=1):
    logging.info(f'Loading {input_file}, channel {channel}')
    img = np.load(input_file, mmap_mode='c')['probabilities'][channel]  # zyx
    logging.info(f'Loaded {input_file}, channel {channel}, shape: {img.shape}')
    return img

# run two-pass watershed on all boundary images within one folder
def run_watershed(path_prediction_in,
                  path_ws_out, boundary_threshold=0.5,
                  sigma_seeds=2.0,
                  export_non_pmap_supervoxels=True,
                  skip_if_ws_exists=True,
                  n_threads=None,
                  min_size=100,
                  # compactness=0,
                  ):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    logging.info(f'Running watershed on {path_prediction_in}')

    if skip_if_ws_exists and os.path.exists(path_ws_out):
        logging.info(f'Skipping {path_prediction_in} because {path_ws_out} exists')
        return

    dirname_out = dirname(path_ws_out)
    if dirname_out != '' and not os.path.exists(dirname_out):
        logging.info(f'Creating directory {dirname_out}')
        os.mkdir(dirname_out)

    file_ending_out = os.path.splitext(os.path.basename(path_ws_out))[-1]
    #detect nii.gz
    if file_ending_out == '.gz':
        if '.nii.gz' in path_ws_out:
            file_ending_out = '.nii.gz'
    logging.info(f'File ending for output: {file_ending_out}')


    logging.info(f'Loading boundary map from {path_prediction_in}')
    boundary_input_zyx = itk.GetArrayFromImage(itk.imread(path_prediction_in))
    logging.info(f'Boundary map shape: {boundary_input_zyx.shape}')

    # swap 0 with 2 if larger shape
    swapped_axes = False
    if boundary_input_zyx.shape[0] > boundary_input_zyx.shape[2]:
        logging.info('Swapping axes 0 and 2')
        logging.info(f'Old shape: {boundary_input_zyx.shape}')
        boundary_input_zyx = np.swapaxes(boundary_input_zyx, 0, 2).copy()
        logging.info(f'New shape: {boundary_input_zyx.shape}')
        swapped_axes = True


    if boundary_input_zyx.max() > 1:
        logging.info('Normalizing boundary map to [0, 1]')
        boundary_input_zyx = (boundary_input_zyx / 255).astype(np.float32)
    else:
        logging.info('Boundary map already in [0, 1]')
        boundary_input_zyx = boundary_input_zyx.astype(np.float32)

    if n_threads is None:
        n_threads = os.cpu_count()
    logging.info(f'Running watershed with threshold {boundary_threshold}'
                 f' and sigma seeds {sigma_seeds} and {n_threads} threads and min size {min_size}')
    watershed = calculate_two_pass_watershed(boundary_input_zyx, boundary_threshold, sigma_seeds=sigma_seeds,
                                             n_threads=n_threads, min_size=min_size,
                                             # compactness=compactness,
                                             )
    if swapped_axes:
        logging.info('Swapping axes 0 and 2 back')
        logging.info(f'shape before swap: {watershed.shape}')
        watershed = np.swapaxes(watershed, 0, 2).copy()
        boundary_input_zyx = np.swapaxes(boundary_input_zyx, 0, 2).copy()
        logging.info(f'shape after swap: {watershed.shape}')


    logging.info('Relabeling watershed')
    watershed = relabel_sequential(watershed)[0]
    watershed = convert_labels_to_smaller_dtype(watershed)

    if export_non_pmap_supervoxels:
        watershed = itk.GetImageFromArray(watershed)
        path_ws_out_without_pmap = path_ws_out.replace(file_ending_out, f'_without_pmap{file_ending_out}')
        logging.info(f'Writing watershed without pmap to {path_ws_out_without_pmap}')
        itk.imwrite(watershed, path_ws_out_without_pmap, compression=True)
        watershed = itk.GetArrayFromImage(watershed)

    logging.info('Setting background to 0')
    watershed = set_threshold_img_to_bg(watershed, boundary_input_zyx, threshold=boundary_threshold)
    watershed = itk.GetImageFromArray(watershed)
    logging.info(f'Writing watershed to {path_ws_out}')
    itk.imwrite(watershed, path_ws_out, compression=True)


def convert_nrrd_to_nnu_nifty(path_input, path_output):
    img = itk.imread(path_input)
    logging.info(f'Converting {path_input} to {path_output}')
    largest_region = img.GetLargestPossibleRegion()
    dim_x, dim_y, dim_z = largest_region.GetSize()
    logging.info(f'Image size (x, y, z): {dim_x}, {dim_y}, {dim_z} -- make sure this it is shown as xyz!')
    # swap!
    img = np.swapaxes(itk.GetArrayFromImage(img), 0, 2)
    img = itk.GetImageFromArray(img)
    itk.imwrite(img, path_output)


# mask_id=self.mask_id,
def run_nnu(input_nifty, output_folder, dataset_id, disable_tta=True, folds=[0, 1, 2, 3, 4], step_size=0.5, plan='nnUNetPlans'):
    # dataset_id: 821 bnd, 822 mask
    # assert dataset_id in [821, 822], 'dataset_id has to be 821 or 822'

    assert os.path.exists(input_nifty), f'File not found: {input_nifty}'
    # ass nifty nii gz
    assert input_nifty.endswith('.nii.gz'), f'File must end with .nii.gz'

    input_folder = os.path.dirname(input_nifty)

    skip_prediction_if_files_exist = True
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if folds == [0, 1, 2, 3, 4]:
        # all folds, fold command can be ommited
        fold_cmd = ' '
    else:
        fold_cmd = ' -f '
        for fold in folds:
            fold_cmd += f' {fold}'
        fold_cmd += ' '
    if not plan:
        plan_cmd = ''
    else:
        plan_cmd = f' -p {plan} '

    cmd = (f'nnUNetv2_predict -i {input_folder} -o {output_folder} {fold_cmd} {plan_cmd}'
           f'-c 3d_fullres -d {dataset_id} --verbose --save_probabilities -step_size {step_size}')

    if disable_tta:
        cmd += ' --disable_tta'

    if skip_prediction_if_files_exist:
        cmd += ' --continue_prediction'

    logging.info(f'Running command: {cmd}')
    subprocess.run(cmd, shell=True, check=True)


def convert_nnu_bnd_to_nrrd(input_file, output_file, skip_if_output_exists=True):
    if not os.path.exists(input_file):
        logging.warning(f'File not found: {input_file}')
        return
    if skip_if_output_exists and os.path.exists(output_file):
        logging.info(f'Skipping {output_file} as it already exists')
        return
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file))

    bnd = read_nnu_proba(input_file, channel=1)
    itk.imwrite(itk.GetImageFromArray(bnd), output_file, compression=True)
    logging.info(f'Saved {output_file}')


def combined_bnd_and_cm_mask(folder_bnd, folder_mask, output_folder, skip_if_combined_file_exists=True):
    # glob npz file bnd
    npz_files_bnd = glob.glob(os.path.join(folder_bnd, '*.npz'))
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for npz_file_bnd in npz_files_bnd:
        stackname = os.path.splitext(os.path.basename(npz_file_bnd))[0]
        npz_file_mask = npz_file_bnd.replace(folder_bnd, folder_mask)
        if not os.path.exists(npz_file_mask):
            logging.warning(f'File not found: {npz_file_mask}')
            continue

        output_filename = f'{stackname}.nrrd'
        output_file = os.path.join(output_folder, output_filename)
        if skip_if_combined_file_exists and os.path.exists(output_file):
            logging.info(f'Skipping {output_file} as it already exists')
            continue

        # load bnd
        bnd = read_nnu_proba(npz_file_bnd, channel=1)
        mask = read_nnu_proba(npz_file_mask, channel=0)

        # import napari
        # viewer = napari.view_image(bnd)
        # viewer.add_image(mask)
        # napari.run()

        # combine both
        combined = np.max((bnd, mask), axis=0)

        # import napari
        # viewer = napari.view_image(combined)
        # napari.run()

        itk.imwrite(itk.GetImageFromArray(combined), output_file, compression=True)
        logging.info(f'Saved {output_file}')


def run_freiburg_mc(input_file, folder_output=None,
                    betas=[0.075],
                    betas2=[],
                    relabel_seg=True,
                    bnd_dataset_id=821,
                    mask_dataset_id=822,
                    disable_tta=True,
                    step_size=0.5,
                    n_threads=None,
                    min_size=100,
                    # compactness=0,
                    folds_bnd=[0,1,2,3,4],
                    folds_mask=[0,1,2,3,4],
                    plan_bnd='nnUNetPlans',
                    plan_mask='nnUNetPlans',
                    run_bnd_only_ws=False,
                    ):
    if folder_output is None:
        folder_output = os.path.dirname(input_file)
    if not os.path.exists(folder_output):
        os.makedirs(folder_output)
    if isinstance(folds_bnd, str):
        if ',' in folds_bnd:
            folds_bnd = [f for f in folds_bnd.split(',')]
        else:
            folds_bnd = [folds_bnd]
    if isinstance(folds_mask, str):
        if ',' in folds_mask:
            folds_mask = [f for f in folds_mask.split(',')]
        else:
            folds_mask = [folds_mask]


    filename = os.path.basename(input_file)
    file_ext = os.path.splitext(filename)[-1]
    if file_ext == '.gz':
        if '.nii.gz' in filename:
            file_ext = '.nii.gz'
    stackname = os.path.basename(input_file).replace(file_ext, '')

    folder_bnd = os.path.join(folder_output, f'prediction_{bnd_dataset_id}')
    folder_mask = os.path.join(folder_output, f'prediction_{mask_dataset_id}')
    folder_combined = os.path.join(folder_output, 'prediction_combined')

    filename_nifty = os.path.basename(input_file).replace(file_ext, '_0000.nii.gz')
    input_file_nifty = os.path.join(folder_output, filename_nifty)
    if not os.path.exists(input_file_nifty):
        convert_nrrd_to_nnu_nifty(input_file, input_file_nifty)

    # first: predict bnd
    run_nnu(input_nifty=input_file_nifty, output_folder=folder_bnd, dataset_id=bnd_dataset_id, folds=folds_bnd, disable_tta=disable_tta, step_size=step_size,
            plan=plan_bnd)
    # predict mask
    run_nnu(input_nifty=input_file_nifty, output_folder=folder_mask, dataset_id=mask_dataset_id, folds=folds_mask, disable_tta=disable_tta, step_size=step_size,
            plan=plan_mask)

    if run_bnd_only_ws:
        path_to_nnu_bnd_npz = os.path.join(folder_bnd, f'{stackname}.npz')
        output_path_bnd = os.path.join(folder_bnd, f'{stackname}.nrrd')
        convert_nnu_bnd_to_nrrd(path_to_nnu_bnd_npz, output_path_bnd)
        output_path_bnd_ws = os.path.join(folder_bnd, f'{stackname}_ws.nrrd')
        input_ws_bnd_pmap = os.path.join(folder_bnd, f'{stackname}_ws_without_pmap.nrrd')
        output_mc_folder_bnd = os.path.join(folder_bnd, f'{stackname}_MultiCut')
        run_watershed(path_prediction_in=output_path_bnd, path_ws_out=output_path_bnd_ws, n_threads=n_threads, min_size=min_size)
        run_multicut(path_to_prediction=output_path_bnd, path_to_watershed=input_ws_bnd_pmap, output_folder=output_mc_folder_bnd, betas=betas)

    # merge both
    combined_bnd_and_cm_mask(folder_bnd, folder_mask, folder_combined)

    # run ws
    input_ws_boundary = os.path.join(folder_combined, f'{stackname}.nrrd')
    output_ws_boundary = os.path.join(folder_combined, f'{stackname}_ws.nrrd')
    run_watershed(input_ws_boundary, output_ws_boundary, n_threads=n_threads, min_size=min_size,
                  # compactness=compactness
                  )

    # run mc
    input_mc = os.path.join(folder_combined, f'{stackname}.nrrd')
    input_ws = os.path.join(folder_combined, f'{stackname}_ws_without_pmap.nrrd')
    output_mc_folder = os.path.join(folder_combined, f'{stackname}_MultiCut')
    run_multicut(input_mc, input_ws, output_mc_folder, betas=betas)

    # run 2nd multicut, but on the multicut itself
    # all_mcs = glob.glob(os.path.join(output_mc_folder, f'*_pmap_zero.nrrd'))
    # for mc in tqdm(all_mcs):
    #     stackname = os.path.basename(mc).replace('_pmap_zero.nrrd', '')
    #     output_folder = os.path.join(output_mc_folder, f'{stackname}_MultiCut2')
    #     run_multicut(path_to_prediction=input_mc, path_to_watershed=mc, output_folder=output_folder, betas=betas2,
    #                  relabel_seg=relabel_seg)


if __name__  == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # input_file = '/mnt/work/data/tmp/segmentation/human/imgs_norm/test_end_to_end/160130_2.nrrd'
    input_file = '/mnt/work/data/tmp/segmentation/human/tof/original_pipeline/zStackCARE2_not_swapped.nrrd'
    input_file = '/mnt/filosaurus/data/confocal/mechanical_waves/img_stack/deblurred/crop/2023_11_09_Drift.lif - Series011_att_corrected-cropped_1_700.nrrd'
    input_file = '/mnt/work/tmp/seg_stack_2/2024_11_08_WGA_Rabbit_stack_1_wga.nrrd'
    run_freiburg_mc(input_file)
