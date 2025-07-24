from typing import Union
import SimpleITK as sitk
import numpy as np


def check_if_bounding_box_touches_border(bounding_box, image_dimensions, buffer=3):
    # check if bounding box touches border within buffer
    # bounding box is  [xstart, ystart, zstart, xsize, ysize, zsize]
    # image dimensions is [xsize, ysize, zsize]
    xstart, ystart, zstart, xsize, ysize, zsize = bounding_box
    xend = xstart + xsize
    yend = ystart + ysize
    zend = zstart + zsize

    # check if start is within buffer
    if any([xstart <= buffer,
            ystart <= buffer,
            zstart <= buffer]):
        return True

    # check if end is near border
    if any([xend > image_dimensions[0] - buffer,
            yend > image_dimensions[1] - buffer,
            zend > image_dimensions[2] - buffer]):
        return True

    return False

def is_on_border(bounding_box, size_volume, buffer=30):
    #check if bounding box touches border
    # (within buffer, i.e. if buffer is 30, then if bounding box is within 30 pixels of the border)

    is_touching_border = 0
    # if bounding_box[0] == 0 or bounding_box[1] == 0 or bounding_box[2] == 0:
    if bounding_box[0] < buffer or bounding_box[1] < buffer or bounding_box[2] < buffer:
        is_touching_border = 1
    # if bounding_box[3] == size_volume[0] or bounding_box[4] == size_volume[1] or bounding_box[5] == size_volume[2]:
    if (
            bounding_box[0] + bounding_box[3] > size_volume[0] - buffer or
            bounding_box[1] + bounding_box[4] > size_volume[1] - buffer or
            bounding_box[2] + bounding_box[5] > size_volume[2] - buffer
    ):
        is_touching_border = 1

    return is_touching_border

def binarize_is_equal_to_label(image: sitk.Image, label: int, fill_label: int = 1) -> sitk.Image:
    # Binarize the image such that all pixels with the value 'label' are set to 'fill_label', rest are set to 0.
    assert isinstance(image, sitk.Image), "image must be a SimpleITK image"

    binary_filter = sitk.BinaryThresholdImageFilter()
    binary_filter.SetLowerThreshold(label)
    binary_filter.SetUpperThreshold(label)

    binary_filter.SetInsideValue(fill_label)  # Value for the 'label' pixels
    binary_filter.SetOutsideValue(0)  # Value for the other pixels

    binary_image = binary_filter.Execute(image)
    return binary_image

def align_with_pca(mask: Union[sitk.Image, np.ndarray], image: Union[sitk.Image, np.ndarray] = None, label: int = 255, buffer: int = 30,
                   mask_interpolator=sitk.sitkNearestNeighbor,
                   wga_interpolator=sitk.sitkLinear,
                   visualise: bool = True,
                   verbose: bool = False,
                   binarize: bool = False,
                   border_touching_buffer=3,
                     mask2: Union[sitk.Image, np.ndarray] = None) -> Union[sitk.Image, tuple]:
    # resample mask to aligned coordinate system using PCA

    info_dict = {}

    if type(mask) is np.ndarray:
        mask = sitk.GetImageFromArray(mask)

    if type(mask2) is np.ndarray:
        mask2 = sitk.GetImageFromArray(mask2)

    if image is not None:
        if type(image) is np.ndarray:
            image = sitk.GetImageFromArray(image)
        if type(image) is not sitk.SimpleITK.Image:
            raise ValueError("Image is not sitk image, something in IO went wrong?")

    if type(mask) is not sitk.SimpleITK.Image:
        raise ValueError("Image is not sitk image, something in IO went wrong?")


    shape_stats = sitk.LabelShapeStatisticsImageFilter()
    shape_stats.ComputeOrientedBoundingBoxOn()
    shape_stats.Execute(mask)

    bounding_box = shape_stats.GetBoundingBox(label)
    mask_dimensions = mask.GetSize()
    is_touching_border = check_if_bounding_box_touches_border(bounding_box, mask_dimensions, buffer=border_touching_buffer)
    info_dict['is_touching_border'] = is_touching_border
    info_dict['border_touching_buffer'] = border_touching_buffer

    xstart, ystart, zstart, xsize, ysize, zsize = bounding_box
    xend = xstart + xsize
    yend = ystart + ysize
    zend = zstart + zsize
    info_dict['bb_xstart'] = xstart
    info_dict['bb_ystart'] = ystart
    info_dict['bb_zstart'] = zstart
    info_dict['bb_xend'] = xend
    info_dict['bb_yend'] = yend
    info_dict['bb_zend'] = zend

    # GetOrientedBoundingBoxSize returns sizes in physical units, not index space, itk uses x y z conversion
    # important that resolution is set correctly in the mask!
    mask_oriented_bounding_box_size = shape_stats.GetOrientedBoundingBoxSize(label)
    info_dict['cm_bounding_box_size_x_microm'] = f'{1e-3 * mask_oriented_bounding_box_size[0]:.3f}'
    info_dict['cm_bounding_box_size_y_microm'] = f'{1e-3 * mask_oriented_bounding_box_size[1]:.3f}'
    info_dict['cm_bounding_box_size_z_microm'] = f'{1e-3 * mask_oriented_bounding_box_size[2]:.3f}'

    centroid = shape_stats.GetCentroid(label)
    info_dict['centroid_x'] = f'{centroid[0]:.1f}'
    info_dict['centroid_y'] = f'{centroid[1]:.1f}'
    info_dict['centroid_z'] = f'{centroid[2]:.1f}'

    if verbose:
        print(f'OrientedBoundingBoxSize: {mask_oriented_bounding_box_size}')

    min_spacing = min(mask.GetSpacing())
    aligned_image_spacing = [min_spacing, min_spacing, min_spacing]

    aligned_image_size_index_space = list()
    for i in range(3):
        aligned_image_size_index_space.append(int(np.ceil(shape_stats.GetOrientedBoundingBoxSize(label)[i] /
                                                          aligned_image_spacing[i])))
    if verbose:
        print(f'aligned image size (index space): {aligned_image_size_index_space}')

    direction_mat = shape_stats.GetOrientedBoundingBoxDirection(label)
    aligned_image_direction = [direction_mat[0], direction_mat[3], direction_mat[6],
                               direction_mat[1], direction_mat[4], direction_mat[7],
                               direction_mat[2], direction_mat[5], direction_mat[8]]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputDirection(aligned_image_direction)

    # Unit vectors in new coordinate system
    mat_first = -np.array((direction_mat[0], direction_mat[1], direction_mat[2]))
    mat_second = -np.array((direction_mat[3], direction_mat[4], direction_mat[5]))
    mat_third = -np.array((direction_mat[6], direction_mat[7], direction_mat[8]))

    info_dict['orientation_vec_first'] = f'{mat_first[0]:.2f} {mat_first[1]:.2f} {mat_first[2]:.2f}'
    info_dict['orientation_vec_second'] = f'{mat_second[0]:.2f} {mat_second[1]:.2f} {mat_second[2]:.2f}'
    info_dict['orientation_vec_third'] = f'{mat_third[0]:.2f} {mat_third[1]:.2f} {mat_third[2]:.2f}'

    if verbose:
        print('-1 * aligned_image_direction:')
        print(info_dict['orientation_vec_first'])
        print(info_dict['orientation_vec_second'])
        print(info_dict['orientation_vec_third'])

    # Buffer(currentLabelId) = Unit vector(currentLabelId) * predefined Buffer_length
    buffer_first = mat_first * buffer
    buffer_second = mat_second * buffer
    buffer_third = mat_third * buffer

    # Defining new Origin with Buffer
    original_origin = np.array(shape_stats.GetOrientedBoundingBoxOrigin(label))
    origin_with_buffer = buffer_first + buffer_second + buffer_third + original_origin
    resampler.SetOutputOrigin(origin_with_buffer)

    if verbose:
        print(f'Origin of aligned coordinate system: {original_origin}')
        print(f'3D difference buffer/BB: {(origin_with_buffer - original_origin)}')

    # Defining new Image size with buffer
    resampler.SetOutputSpacing(aligned_image_spacing)
    aligned_image_size_with_buffer = [n + 2 * buffer for n in aligned_image_size_index_space] # fix buffer
    resampler.SetSize(aligned_image_size_with_buffer)

    # Mask interpolated with Nearest Neighbour interpolation (since mask is a binary image)
    resampler.SetInterpolator(mask_interpolator)

    rotated_mask_sitk = resampler.Execute(mask)

    # WGA interpolated with linear interpolation (since image is not binary)
    if image is not None:
        resampler.SetInterpolator(wga_interpolator)
        rotated_wga_sitk = resampler.Execute(image)

    if mask2 is not None:
        resampler.SetInterpolator(mask_interpolator)
        rotated_mask2_sitk = resampler.Execute(mask2)

    if binarize:
        # set other labels to 0, set label itself to true
        rotated_mask_sitk = binarize_is_equal_to_label(image=rotated_mask_sitk, label=label, fill_label=1)
        rotated_mask_sitk = sitk.Cast(rotated_mask_sitk, sitk.sitkUInt8)
    else:
        # set other labels to 0, keep the original label
        rotated_mask_sitk = binarize_is_equal_to_label(image=rotated_mask_sitk, label=label, fill_label=label)

    if visualise:
        import napari
        viewer = napari.Viewer()
        # viewer.add_image(image_mask, name='cell mask', colormap= 'PiYG')
        if image is not None:
            viewer.add_image(sitk.GetArrayFromImage(rotated_wga_sitk), name='rotated image')
        viewer.add_labels(sitk.GetArrayFromImage(rotated_mask_sitk), name='rerotated_image')
        napari.run()

    if mask2 is None:
        if image is None:
            return rotated_mask_sitk, info_dict,
        else:
            return rotated_mask_sitk, rotated_wga_sitk, mask_oriented_bounding_box_size, info_dict
    else:
        if image is None:
            return rotated_mask_sitk, info_dict, rotated_mask2_sitk
        else:
            return rotated_mask_sitk, rotated_wga_sitk, mask_oriented_bounding_box_size, info_dict, rotated_mask2_sitk