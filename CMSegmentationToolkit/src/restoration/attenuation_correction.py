import logging
import itk
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import maximum_filter1d
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from tqdm import tqdm


def get_valid_slices(image, threshold_z, min_fraction_for_valid=0.2):
    logging.info(f'calculating valid slices')
    valid_slices = np.ones(image.shape[0], dtype=bool)
    for z in range(0, image.shape[0]):
        if np.mean(image[z, :] > threshold_z[z]) < min_fraction_for_valid:
            valid_slices[z] = False

    fraction_invalid_slices = 1 - np.sum(valid_slices) / len(valid_slices)
    if fraction_invalid_slices > 0.5:
        logging.warning(f'Fraction of invalid slices is {fraction_invalid_slices}, returning all valid')
        valid_slices = np.ones(image.shape[0], dtype=bool)
        for z in range(0, image.shape[0]):
            if np.mean(image[z, :] > threshold_z[z]) < min_fraction_for_valid/2:
                valid_slices[z] = False
    logging.info(f'valid slices calculated')
    return valid_slices

def get_adaptive_threshold_z(image):
    logging.info(f'calculating adaptive threshold')
    q90 = np.quantile(image, axis=(1,2), q=0.9)
    filterlength = 150 if len(q90) > 150 else len(q90)
    q90_max_filtered = maximum_filter1d(q90, filterlength)
    threshold_z = 0.1*q90_max_filtered
    logging.info(f'adaptive threshold calculated')
    return threshold_z

def fill_zeros_with_nearest_nonzero(array):
    non_zero_indices = np.nonzero(array)[0]
    if len(non_zero_indices) == 0:
        return array
    non_zero_values = array[non_zero_indices]
    extrapolation_function = interp1d(non_zero_indices, non_zero_values, kind='nearest', fill_value='extrapolate')
    filled_array = extrapolation_function(np.arange(len(array)))
    return filled_array


def subtract_background_2D(image, radius=50):
    from pyclesperanto_prototype import top_hat_sphere
    from pyclesperanto_prototype import push
    # conda install -c conda-forge ocl-icd-system
    # conda install -c conda-forge pyclesperanto-prototype


    # decompose in 2D slice along z, apply rolling ball, recompose
    result_volume = np.zeros_like(image)
    for z in tqdm(range(image.shape[0]), desc='top hat filtering'):
        result_image = None
        result_image = top_hat_sphere(source=push(image[z]), destination=result_image, radius_x=radius, radius_y=radius)
        result_volume[z] = result_image
    logging.info(f'top hat filtering done.')
    return result_volume

def attenuation_correction(img_in, path_image_out=None, debug=False):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    # assumption, zyx
    logging.info(f'writing to {path_image_out}')
    logging.info(f'debug: {debug}')

    if isinstance(img_in, str):
        logging.info(f'loading {img_in}')
        image = itk.imread(img_in)
        image = itk.GetArrayFromImage(image)
        image = image.astype(np.float32)
    elif isinstance(img_in, np.ndarray):
        image = img_in.astype(np.float32)
    logging.info(f'shape: {image.shape} (should be zyx)')

    # image_original = image.copy()
    image_bg_subtracted = image.copy()
    image_bg_subtracted = subtract_background_2D(image_bg_subtracted, radius=25)
    image_bg = image - image_bg_subtracted

    max_value_dtype = np.iinfo(np.uint16).max
    norm_value_background = 0.05 * max_value_dtype
    norm_value_fg = 0.3 * max_value_dtype

    background_estimate = np.quantile(image_bg, axis=(1,2), q=0.01)
    foreground_estimate = background_estimate + np.quantile(image_bg_subtracted, axis=(1,2), q=0.99)

    # smooth both estimates, check that filterlength fits, has to be uneven
    filterlength = 101 if len(background_estimate) > 101 else len(background_estimate)
    filterlength = filterlength if filterlength % 2 == 1 else filterlength - 1
    background_estimate = savgol_filter(background_estimate, filterlength, 3)
    foreground_estimate = savgol_filter(foreground_estimate, filterlength, 3)

    threshold_z = get_adaptive_threshold_z(image_bg_subtracted)
    valid_indices = get_valid_slices(image_bg_subtracted, threshold_z=threshold_z)

    background_estimate[valid_indices == 0] = 0
    foreground_estimate[valid_indices == 0] = 0
    background_estimate = fill_zeros_with_nearest_nonzero(background_estimate)
    foreground_estimate = fill_zeros_with_nearest_nonzero(foreground_estimate)

    image_corrected = image.copy()
    for z in tqdm(range(image.shape[0]), desc='normalising image'):
        image_corrected[z] = (image_corrected[z] - background_estimate[z]) # background = 0
        denominator = foreground_estimate[z] - background_estimate[z] + 1e-6
        image_corrected[z] = image_corrected[z] / denominator  # foreground = 1
        image_corrected[z] = image_corrected[z] * (norm_value_fg - norm_value_background) # fg = norm_fg - norm_bg
        image_corrected[z] = image_corrected[z] + norm_value_background # bg = norm_bg; fg = norm_fg

    if debug:
        logging.info('plotting')
        # plot histogram and clip borders
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.hist(image_corrected.flatten(), bins=512, alpha=0.5, label='corrected')
        plt.axvline(norm_value_background, color='r', linestyle='--', label='background')
        plt.axvline(norm_value_fg, color='g', linestyle='--', label='foreground')
        plt.axvline(0, color='k', linestyle='--', label='0')
        plt.axvline(max_value_dtype, color='k', linestyle='--', label='max dtype')
        plt.legend()
        ax1 = plt.subplot(1, 3, 2)
        plt.plot(valid_indices, color='b', label='valid indices')
        plt.legend(loc='upper right')
        ax2 = ax1.twinx()
        ax2.plot(threshold_z, color='r', label='threshold')
        plt.legend('upper left')
        ax3 = plt.subplot(1, 3, 3)
        plt.plot(background_estimate, label='background estimate')
        plt.legend(loc='upper right')
        ax4 = ax3.twinx()
        plt.plot(foreground_estimate, label='foreground estimate')
        plt.legend(loc='upper left')
        plt.show(block=True)
        logging.info('plotting done')


    image_corrected = np.clip(image_corrected, 0, max_value_dtype).astype(np.uint16)

    if debug:
        logging.info('napari visualisation')
        import napari
        thresholded_wga = np.zeros_like(image_bg_subtracted)
        for z in range(image.shape[0]):
            thresholded_wga[z] = image_bg_subtracted[z] > threshold_z[z]
        viewer = napari.view_image(image, name='input_array', colormap='inferno')
        viewer.add_image(image_corrected, name='image_corrected', colormap='inferno')
        viewer.add_image(image_bg_subtracted, name='image_bg_subtracted', colormap='inferno')
        viewer.add_image(thresholded_wga, name='thresholded_wga', colormap='inferno')
        napari.run()
        logging.info('napari visualisation done')

    if path_image_out is not None:
        logging.info('writing image')
        itk.imwrite(itk.GetImageFromArray(image_corrected), path_image_out)
        logging.info('writing image done')
    else:
        return image_corrected
