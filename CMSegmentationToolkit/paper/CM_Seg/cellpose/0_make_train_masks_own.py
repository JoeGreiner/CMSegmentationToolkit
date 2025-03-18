import os
import glob
import numpy as np
import tifffile
import tqdm
import logging
import itk
import shutil

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    path_to_base_folder = '/mnt/work/data/CM_Seg_Paper_24/data_v2'
    output_folder = '/mnt/work/data/CM_Seg_Paper_24/cellpose'
    output_folder_train = os.path.join(output_folder, 'train')
    output_folder_val = os.path.join(output_folder, 'val')
    output_folder_test = os.path.join(output_folder, 'test')

    for folder in [output_folder_train, output_folder_val, output_folder_test]:
        if os.path.exists(folder):
            logging.info(f"deleting {folder}")
            shutil.rmtree(folder)

        if not os.path.exists(folder):
            os.makedirs(folder)

    n_crops_per_image = 20
    min_crop_size = 100

    crop_size_YX = [512, 512]
    crop_size_ortho = [min_crop_size, 512]


    all_train_and_val_imgs_nrrds = glob.glob(os.path.join(path_to_base_folder, "*", "train", 'imgs_norm', '*.nrrd'),
                                             recursive=True)
    all_test_imgs_nrrds = glob.glob(os.path.join(path_to_base_folder, "*", "test", 'imgs_norm', '*.nrrd'),
                                    recursive=True)

    assert len(all_train_and_val_imgs_nrrds) > 0, 'Could not find any training images'
    assert len(all_test_imgs_nrrds) > 0, 'Could not find any test images'

    # seed 42
    np.random.seed(42)
    val_fraction = 0.2
    np.random.shuffle(all_train_and_val_imgs_nrrds)
    n_val = int(val_fraction * len(all_train_and_val_imgs_nrrds))
    all_train_imgs_nrrds = all_train_and_val_imgs_nrrds[n_val:]
    all_val_imgs_nrrds = all_train_and_val_imgs_nrrds[:n_val]

    logging.info(f'{len(all_train_imgs_nrrds)} training images')
    logging.info(f'{len(all_val_imgs_nrrds)} validation images')
    logging.info(f'{len(all_test_imgs_nrrds)} test images')

    for img_list, output_folder in zip([all_train_imgs_nrrds, all_val_imgs_nrrds, all_test_imgs_nrrds],
                                       [output_folder_train, output_folder_val, output_folder_test]):
        for img_path in tqdm.tqdm(img_list):
            path_to_instance_seg = img_path.replace('imgs_norm', 'labels')
            assert os.path.exists(path_to_instance_seg), f'Could not find {path_to_instance_seg}'

            img0 = itk.GetArrayFromImage(itk.imread(img_path))
            mask0 = itk.GetArrayFromImage(itk.imread(path_to_instance_seg))

            assert img0.shape == mask0.shape, f"image shape {img0.shape} does not match mask shape {mask0.shape}"
            dimZ, dimY, dimX = img0.shape

            # YX
            output_folder_xy = os.path.join(output_folder, 'xy')
            if not os.path.exists(output_folder_xy):
                os.makedirs(output_folder_xy)
            if dimY <= min_crop_size or dimX <= min_crop_size:
                print(f"skipping {img_path} because it has less than {min_crop_size} slices")
            else:
                # draw random crops in z first
                z_indices_drawn = np.random.choice(dimZ, n_crops_per_image, replace=True)
                logging.info(f"z_indices_drawn: {z_indices_drawn}")
                for z in z_indices_drawn:
                    img_z = img0[z]
                    mask_z = mask0[z]

                    crop_size_y, crop_size_x = crop_size_YX
                    assert dimY >= crop_size_y, f"dimY {dimY} smaller than crop_size_y {crop_size_y}"
                    assert dimX >= crop_size_x, f"dimX {dimX} smaller than crop_size_x {crop_size_x}"

                    # crop random roi
                    y = np.random.randint(0, dimY - crop_size_y)
                    x = np.random.randint(0, dimX - crop_size_x)
                    logging.info(f"cropping {img_path} at z={z} y={y} x={x}")

                    img_z_crop = img_z[y:y + crop_size_y, x:x + crop_size_x]
                    mask_z_crop = mask_z[y:y + crop_size_y, x:x + crop_size_x]

                    filename_no_ext = os.path.splitext(os.path.basename(img_path))[0]
                    output_img = os.path.join(output_folder_xy, f'{filename_no_ext}_{z}.tif')
                    output_mask = os.path.join(output_folder_xy,
                                               f'{filename_no_ext}_{z}_masks.tif')

                    tifffile.imwrite(output_img, img_z_crop)
                    tifffile.imwrite(output_mask, mask_z_crop)

            # ZY
            output_folder_zy = os.path.join(output_folder, 'ortho')
            if not os.path.exists(output_folder_zy):
                os.makedirs(output_folder_zy)

            if dimZ <= min_crop_size or dimY <= min_crop_size:
                print(f"skipping {img_path} because it has less than {min_crop_size} slices")
            else:
                # draw random crops in x first
                x_indices_drawn = np.random.choice(dimX, n_crops_per_image, replace=True)
                logging.info(f"x_indices_drawn: {x_indices_drawn}")
                for x in x_indices_drawn:
                    img_x = img0[:, :, x]
                    mask_x = mask0[:, :, x]

                    crop_size_z, crop_size_y = crop_size_ortho
                    assert dimZ >= crop_size_z, f"dimZ {dimZ} smaller than crop_size_z {crop_size_z}"
                    assert dimY >= crop_size_y, f"dimY {dimY} smaller than crop_size_y {crop_size_y}"

                    # crop random roi
                    z = np.random.randint(0, dimZ - crop_size_z)
                    y = np.random.randint(0, dimY - crop_size_y)
                    logging.info(f"cropping {img_path} at z={z} y={y} x={x}")

                    img_x_crop = img_x[z:z + crop_size_z, y:y + crop_size_y]
                    mask_x_crop = mask_x[z:z + crop_size_z, y:y + crop_size_y]

                    filename_no_ext = os.path.splitext(os.path.basename(img_path))[0]
                    output_img = os.path.join(output_folder_zy, f'{filename_no_ext}_zy_{x}.tif')
                    output_mask = os.path.join(output_folder_zy,
                                               f'{filename_no_ext}_zy_{x}_masks.tif')

                    tifffile.imwrite(output_img, img_x_crop)
                    tifffile.imwrite(output_mask, mask_x_crop)


            # ZX
            output_folder_zx = os.path.join(output_folder, 'ortho')
            if not os.path.exists(output_folder_zx):
                os.makedirs(output_folder_zx)

            if dimZ <= min_crop_size or dimX <= min_crop_size:
                print(f"skipping {img_path} because it has less than {min_crop_size} slices")
            else:
                # draw random crops in y first
                y_indices_drawn = np.random.choice(dimY, n_crops_per_image, replace=True)
                logging.info(f"y_indices_drawn: {y_indices_drawn}")
                for y in y_indices_drawn:
                    img_y = img0[:, y, :]
                    mask_y = mask0[:, y, :]

                    crop_size_z, crop_size_x = crop_size_ortho
                    assert dimZ >= crop_size_z, f"dimZ {dimZ} smaller than crop_size_z {crop_size_z}"
                    assert dimX >= crop_size_x, f"dimX {dimX} smaller than crop_size_x {crop_size_x}"

                    # crop random roi
                    z = np.random.randint(0, dimZ - crop_size_z)
                    x = np.random.randint(0, dimX - crop_size_x)
                    logging.info(f"cropping {img_path} at z={z} y={y} x={x}")

                    img_y_crop = img_y[z:z + crop_size_z, x:x + crop_size_x]
                    mask_y_crop = mask_y[z:z + crop_size_z, x:x + crop_size_x]

                    filename_no_ext = os.path.splitext(os.path.basename(img_path))[0]
                    output_img = os.path.join(output_folder_zx, f'{filename_no_ext}_zx_{y}.tif')
                    output_mask = os.path.join(output_folder_zx,
                                               f'{filename_no_ext}_zx_{y}_masks.tif')

                    tifffile.imwrite(output_img, img_y_crop)
                    tifffile.imwrite(output_mask, mask_y_crop)
