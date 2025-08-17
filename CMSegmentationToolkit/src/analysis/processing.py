import os
import SimpleITK as sitk
import numpy as np
from skimage.filters import threshold_li
from tqdm import tqdm
import pandas as pd
import pyclesperanto_prototype as cle
import logging
from CMSegmentationToolkit.src.analysis.transformations import align_with_pca, is_on_border

def threshold_li_modified(img):
    return 0.75 * threshold_li(img)

def white_2d_tophat_pyclesperanto(img, radius=5):
    img = img.astype(np.float32)
    output = cle.create_like(img)
    cle.top_hat_box(source=img, destination=output, radius_x=radius, radius_y=radius, radius_z=1)
    return cle.pull(output)

def analyze_stack(seg, wga=None, resolution_zyx_um=[0.2, 0.2, 0.2], tophat_radius=5,
                  calculate_tats_density=True,
                  debug=False, export_distancemap_path=None, export_distancemap=True,
                  additional_keywords = None, index_prefix="",
                  ):

    logging.info(f"Analyzing stack with parameters: resolution_zyx={resolution_zyx_um}, "
                    f"tophat_radius={tophat_radius}, calculate_tats_density={calculate_tats_density}, "
                    f"debug={debug}, export_path={export_distancemap_path}, export={export_distancemap}")

    if not isinstance(seg, np.ndarray):
        if isinstance(seg, str) or isinstance(seg, os.PathLike):
            seg = sitk.GetArrayFromImage(sitk.ReadImage(seg))
        else:
            assert isinstance(seg, np.ndarray), \
                f"Expected segmentation to be a numpy array or a path to an image, got {type(seg)}"
    assert seg.ndim == 3, "Segmentation must be a 3D array (Z, Y, X)"

    assert isinstance(calculate_tats_density, bool), "calculate_tats_density must be a boolean"
    assert isinstance(debug, bool), "debug must be a boolean"
    assert isinstance(export_distancemap, bool), "export_distancemap must be a boolean"
    assert isinstance(tophat_radius, int) and tophat_radius > 0, "tophat_radius must be a positive integer"

    assert resolution_zyx_um is not None, "resolution_zyx_um must be provided"
    assert isinstance(resolution_zyx_um, (list, tuple)) and len(resolution_zyx_um) == 3, \
        "resolution_zyx_um must be a list or tuple of length 3"

    if additional_keywords is not None:
        assert isinstance(additional_keywords, dict), "Additional keywords must be a dictionary"

    if calculate_tats_density:
        if wga is None:
            logging.warning("No image provided for TATS density calculation. Disabling TATS density calculation.")
            calculate_tats_density = False
        else:
            if not isinstance(wga, np.ndarray):
                if isinstance(wga, str) or isinstance(wga, os.PathLike):
                    wga = sitk.GetArrayFromImage(sitk.ReadImage(wga))
                else:
                    raise TypeError(f"Expected wga to be a numpy array or a path to an image, got {type(wga)}")
            assert wga.ndim == 3, "Image must be a 3D array (Z, Y, X)"
            assert wga.shape == seg.shape, f"Image and segmentation must have the same shape, got {wga.shape} and {seg.shape} for wga and seg respectively"

    myocyte_space_frac = (np.count_nonzero(seg > 0) / np.prod(seg.shape)) * 100

    if wga is not None:
        threshold_li_val = threshold_li_modified(wga[seg > 0])
        extramyocyte_wga_positive = np.zeros_like(seg, dtype=np.uint8)
        extramyocyte_wga_positive[seg == 0] = wga[seg == 0] > threshold_li_val
        number_total_voxels = np.prod(seg.shape)
        extramyocyte_wga_positive_space_frac = np.count_nonzero(extramyocyte_wga_positive) / number_total_voxels * 100

    dim_z, dim_y, dim_x = seg.shape
    if calculate_tats_density:
        for z_ix in tqdm(range(dim_z), desc='Processing z slices'):
            wga[z_ix] = white_2d_tophat_pyclesperanto(wga[z_ix], radius=tophat_radius)  # pyclesperanto is faster

        wga_inside_seg = wga[seg > 0]
        if wga_inside_seg.size == 0:
            print("Warning: No foreground pixels in segmentation. Thresholding might fail.")
            threshold_val = 0
        else:
            threshold_val = threshold_li_modified(wga_inside_seg)
        del wga_inside_seg

        img_itk = sitk.GetImageFromArray(wga)
        img_itk.SetSpacing([resolution_zyx_um[2], resolution_zyx_um[1], resolution_zyx_um[0]])  # here resolution is [res_x, res_y, res_z]
        thresholded_wga = img_itk > threshold_val


        distance_map = sitk.SignedMaurerDistanceMap(thresholded_wga, insideIsPositive=False, squaredDistance=False,
                                                    useImageSpacing=True)



    segmentation_itk = sitk.GetImageFromArray(seg.astype(np.uint32))
    segmentation_itk.SetSpacing([resolution_zyx_um[2], resolution_zyx_um[1], resolution_zyx_um[0]])  # here resolution is [res_x, res_y, res_z]

    border_indicator = get_border_indicator_image(seg)

    if debug and calculate_tats_density:
        import napari
        viewer = napari.Viewer()
        viewer.add_image(sitk.GetArrayFromImage(img_itk), name='Image', scale=resolution_zyx_um[::-1])
        viewer.add_labels(sitk.GetArrayFromImage(thresholded_wga), name='Binary Image', scale=resolution_zyx_um[::-1])
        viewer.add_labels(sitk.GetArrayFromImage(segmentation_itk), name='Segmentation', scale=resolution_zyx_um[::-1])
        viewer.add_image(sitk.GetArrayFromImage(distance_map), name='Distance Map', scale=resolution_zyx_um[::-1])
        napari.run()


    shape_stats_filter = sitk.LabelShapeStatisticsImageFilter()
    shape_stats_filter.ComputeOrientedBoundingBoxOn()
    shape_stats_filter.ComputePerimeterOn()
    shape_stats_filter.Execute(segmentation_itk)
    labels = shape_stats_filter.GetLabels()


    if calculate_tats_density:
        seg[sitk.GetArrayFromImage(thresholded_wga) > 0] = 0 # set to 0 so that only the distance from the cytoplasm is calculated, excluding tats
        segmentation_itk = sitk.GetImageFromArray(seg.astype(np.uint32))
        segmentation_itk.SetSpacing([resolution_zyx_um[2], resolution_zyx_um[1],
                                     resolution_zyx_um[0]])  # here resolution is [res_x, res_y, res_z]

        intensity_stats_filter = sitk.LabelIntensityStatisticsImageFilter()
        intensity_stats_filter.ComputePerimeterOff()
        intensity_stats_filter.SetNumberOfBins(2048) # increase for a more accurate meedian
        # mean is calculated as sum / count, not hist
        intensity_stats_filter.Execute(segmentation_itk, distance_map)

    stats_dict = []

    for label_val in tqdm(labels):
        if label_val == 0: # Skip background label if present
            continue
        bbox = shape_stats_filter.GetBoundingBox(label_val)
        start_x, start_y, start_z, size_x, size_y, size_z = bbox

        assert size_x > 0 and size_y > 0 and size_z > 0, f'Invalid size {size_x} {size_y} {size_z}'
        assert start_x >= 0 and start_y >= 0 and start_z >= 0, f'Invalid start {start_x} {start_y} {start_z}'
        assert (size_z - start_z <= seg.shape[0] and
                size_y - start_y <= seg.shape[1] and
                size_x - start_x <= seg.shape[2]), f'Invalid start {start_x} {start_y} {start_z}'

        mask_cropped = seg[start_z:start_z + size_z, start_y:start_y + size_y, start_x:start_x + size_x].copy()
        mask_cropped = (mask_cropped == label_val).astype(np.uint8)

        border_indicator_cropped = border_indicator[start_z:start_z + size_z, start_y:start_y + size_y, start_x:start_x + size_x].copy()

        rotated_mask_sitk, info_dict, rotated_mask2_sitk = align_with_pca(mask=mask_cropped, label=1, visualise=False, mask2=border_indicator_cropped)
        # rotated_mask_sitk, info_dict = align_with_pca(mask=mask_cropped, label=1, visualise=False)

        avg_second_largest_dim, avg_smallest_dim, avg_area, _ = compute_average_slice_properties(rotated_mask_sitk)

        avg_second_largest_dim_border, avg_smallest_dim_border, avg_area_border, ratio = compute_average_slice_properties(
            rotated_mask_sitk,
            border_indicator=sitk.GetArrayFromImage(rotated_mask2_sitk))


        size_volume = segmentation_itk.GetSize()
        is_touching_border = is_on_border(bbox, size_volume)
        mask_oriented_bounding_box_size = shape_stats_filter.GetOrientedBoundingBoxSize(int(label_val))

        surface_3d_um2 = shape_stats_filter.GetPerimeter(label_val) / (1000 ** 2)  # Convert to um^2

        stats = {
            'index': f"{index_prefix}{label_val}",
            'label': label_val,
            'resolution_x_um': resolution_zyx_um[2],
            'resolution_y_um': resolution_zyx_um[1],
            'resolution_z_um': resolution_zyx_um[0],

            'dimensions_z_voxel': dim_z,
            'dimensions_y_voxel': dim_y,
            'dimensions_x_voxel': dim_x,

            'dimensions_z_um': dim_z * resolution_zyx_um[0],
            'dimensions_y_um': dim_y * resolution_zyx_um[1],
            'dimensions_x_um': dim_x * resolution_zyx_um[2],

            'is_touching_border': is_touching_border,
            'cm_bounding_box_size_0': mask_oriented_bounding_box_size[0],
            'cm_bounding_box_size_1': mask_oriented_bounding_box_size[1],
            'cm_bounding_box_size_2': mask_oriented_bounding_box_size[2],


            'average_second_largest_dim_um': avg_second_largest_dim * resolution_zyx_um[2],
            'average_smallest_dim_um': avg_smallest_dim * resolution_zyx_um[2],
            'average_ratio_dim': avg_second_largest_dim / avg_smallest_dim if avg_smallest_dim > 0 else np.nan,
            'average_area_um2': avg_area * resolution_zyx_um[1] * resolution_zyx_um[2],
            'ratio_width_depth': mask_oriented_bounding_box_size[1] / mask_oriented_bounding_box_size[0],

            'average_second_largest_dim_border_um': avg_second_largest_dim_border * resolution_zyx_um[2],
            'average_smallest_dim_border_um': avg_smallest_dim_border * resolution_zyx_um[2],
            'average_ratio_dim_border': avg_second_largest_dim_border / avg_smallest_dim_border if avg_smallest_dim_border > 0 else np.nan,
            'ratio_skipped_border_slices': ratio,
            'average_area_border_um2': avg_area_border * resolution_zyx_um[1] * resolution_zyx_um[2],

            'cell_volume_um3': shape_stats_filter.GetPhysicalSize(label_val),
            'cell_volume_pL': shape_stats_filter.GetPhysicalSize(label_val) / 1000,
            'cell_volume_voxels': shape_stats_filter.GetNumberOfPixels(label_val),
            'myocyte_space_frac': myocyte_space_frac,

            'flatness_pca': shape_stats_filter.GetFlatness(label_val),
            'boxiness': shape_stats_filter.GetPhysicalSize(label_val) / (mask_oriented_bounding_box_size[0] * mask_oriented_bounding_box_size[1] *
                                      mask_oriented_bounding_box_size[2]),

            'cell_surface_um2' : surface_3d_um2,
            'cell_surface_to_volume_ratio_um': surface_3d_um2 / shape_stats_filter.GetPhysicalSize(label_val)
        }

        if wga is not None:
            stats.update({
                'extramyocyte_wga_positive_space_frac': extramyocyte_wga_positive_space_frac,
                'threshold_li_val': threshold_val
            })

        if calculate_tats_density:
            stats.update({
                'mean': intensity_stats_filter.GetMean(label_val),
                'median': intensity_stats_filter.GetMedian(label_val),
                'std_dev': intensity_stats_filter.GetStandardDeviation(label_val),
                'variance': intensity_stats_filter.GetVariance(label_val),
                'count': intensity_stats_filter.GetNumberOfPixels(label_val),
                'min': intensity_stats_filter.GetMinimum(label_val),
                'max': intensity_stats_filter.GetMaximum(label_val),
                'cell_volume_um3': intensity_stats_filter.GetPhysicalSize(label_val),
                'cell_volume_voxels': intensity_stats_filter.GetNumberOfPixels(label_val)
            })

        stats_dict.append(stats)

    if additional_keywords is not None:
        for stat in stats_dict:
            for key, value in additional_keywords.items():
                stat[key] = value

    df_intensity_stats = pd.DataFrame(stats_dict)
    df_intensity_stats.set_index('index', inplace=True)

    if calculate_tats_density:
        if export_distancemap_path is not None and export_distancemap:
            if not os.path.exists(os.path.dirname(export_distancemap_path)):
                os.makedirs(os.path.dirname(export_distancemap_path))
            # set to 0 outside of the segmentation
            distance_map_array = sitk.GetArrayFromImage(distance_map)
            distance_map_array[seg == 0] = 0
            distance_map = sitk.GetImageFromArray(distance_map_array)
            sitk.WriteImage(distance_map, export_distancemap_path)
            print(f"Distance map exported to {export_distancemap_path}")

    return df_intensity_stats


def get_border_indicator_image(mother_img, border_width = 3):
    assert type(mother_img) is np.ndarray, "mother_img must be a numpy array"
    assert mother_img.ndim == 3, "mother_img must be a 3D array (Z, Y, X)"
    assert border_width > 0, "border_width must be greater than 0"

    logging.info(f"Creating border indicator image with border width {border_width} for image of shape {mother_img.shape}")

    border_indicator = np.zeros_like(mother_img, dtype=np.uint8)
    zDim, yDim, xDim = mother_img.shape
    border_indicator[0:border_width, :, :] = 1  # Top slice
    border_indicator[zDim - border_width:zDim, :, :] = 1  # Bottom slice
    border_indicator[:, 0:border_width, :] = 1  # Left slice
    border_indicator[:, yDim - border_width:yDim, :] = 1  # Right slice
    border_indicator[:, :, 0:border_width] = 1  # Front slice
    border_indicator[:, :, xDim - border_width:xDim] = 1  # Back slice

    return border_indicator



def compute_average_slice_properties(sitk_image, border_indicator=None, area_threshold=50):
    if isinstance(sitk_image, sitk.Image):
        arr = sitk.GetArrayFromImage(sitk_image)
    elif isinstance(sitk_image, np.ndarray):
        arr = sitk_image
    else:
        raise ValueError(f'Unknown type {type(sitk_image)}')

    # axes order: largest dimension, middle dimension, shortest dimension (zyx)

    second_largest_dim_per_length = []
    smallest_dim_per_length = []
    areas = []

    arr = (arr > 0).astype(np.uint8)

    total_z_slices = arr.shape[0]
    skipped_z_slices_border = 0
    for z_index in range(arr.shape[0]):
        slice_ = arr[z_index, :, :]

        if np.any(slice_):

            slice_area = np.count_nonzero(slice_)
            if slice_area < area_threshold:
                continue
            if border_indicator is not None:
                border_indicator_ = border_indicator[z_index, :, :]
                if np.sum(border_indicator_) > 0:
                    skipped_z_slices_border += 1
                    continue

            # Compute the extent of the second largest dimension
            proj_second_largest = np.max(slice_, axis=1)
            second_largest = np.count_nonzero(proj_second_largest)

            # Compute the height in y-direction: max projection along x axis
            proj_smallest_dimension = np.max(slice_, axis=0)
            smallest_dimension = np.count_nonzero(proj_smallest_dimension)

            second_largest_dim_per_length.append(second_largest)
            smallest_dim_per_length.append(smallest_dimension)
            areas.append(slice_area)

    ratio = skipped_z_slices_border / total_z_slices

    debug=False
    if border_indicator is not None:
        if debug:
            if ratio > 0 and ratio < 0.7:
                border_indicator[:, 0, 0] = 2 # small indicator for axis determination

                print(f'Skipped {skipped_z_slices_border} slices due to border indicator,'
                      f' out of {total_z_slices} total slices ({skipped_z_slices_border / total_z_slices * 100:.2f}%)')
                import napari
                viewer = napari.Viewer()
                viewer.add_labels(arr, name='Image', scale=[1, 1, 1])
                viewer.add_labels(border_indicator, name ='Border Indicator', scale=[1, 1, 1])
                napari.run()


    # Calculate averages if we have any valid slices
    avg_second_largest_dim = np.mean(second_largest_dim_per_length) if second_largest_dim_per_length else np.nan
    avg_smallest_dim = np.mean(smallest_dim_per_length) if smallest_dim_per_length else np.nan
    avg_area = np.mean(areas) if areas else np.nan
    if ratio > 0.25:
        avg_second_largest_dim = np.nan
        avg_smallest_dim = np.nan
        avg_area = np.nan

    return avg_second_largest_dim, avg_smallest_dim, avg_area, ratio