import os.path
import shutil
from cellpose import io, models
import glob
import itk
import logging

io.logger_setup()

path_to_base_folder = '/mnt/work/data/CM_Seg_Paper_24/data_v2'
output_folder = '/mnt/work/data/CM_Seg_Paper_24/cellpose'
output_folder_train = os.path.join(output_folder, 'train')
output_folder_val = os.path.join(output_folder, 'val')
output_folder_test = os.path.join(output_folder, 'test')

all_test_imgs_nrrds = glob.glob(os.path.join(path_to_base_folder, "*", "test", 'imgs_norm', '*.nrrd'),
                                    recursive=True)

path_model = '/home/greinerj/PycharmProjects/CM_Seg/cellpose_test/models/XY_WGA_CM'
path_second_model = '/home/greinerj/PycharmProjects/CM_Seg/cellpose_test/models/Ortho_WGA_CM'
model = models.CellposeModel(pretrained_model=path_model, gpu=True)

channels = [[0, 0]]
diameter = 112.808

# Tip from readme: increase flow3D_smooth parameter; tests on val: unfortunately it gets worse for sigma 1,2,3
# 3D segmentation ignores the flow_threshold because we did not find that it helped to filter out false positives in our test 3D cell volume.
flow_sigma = 0

run_one_model_xy_stitch = True
run_two_models_ortho = True
# another tip documentation: use resample=False for faster processing; also need to run it because of OOM errors
resample = False

import datetime
output_folder = '/mnt/work/data/CM_Seg_Paper_24/results'
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
output_folder = os.path.join(output_folder, f'cellpose_test_{timestamp}')
if os.path.exists(output_folder):
    logging.info(f"deleting {output_folder}")
    shutil.rmtree(output_folder)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for path_to_3d_stack in all_test_imgs_nrrds:
    img = itk.GetArrayFromImage(itk.imread(path_to_3d_stack))
    filename_no_extension = os.path.splitext(os.path.basename(path_to_3d_stack))[0]
    fileext_out = 'nrrd'

    # XY application only:  In those instances, you may want to turn off 3D segmentation (do_3D=False) and run instead with stitch_threshold>0.
    if run_one_model_xy_stitch:
        for stitch_threshold in [0.1,]: # I briefly tried tuning, but didn't seem to help much -- default parameters seem to be ok
            for flow_threshold in [0.4]:
                masks, flows, styles = model.eval(img, diameter=diameter, channels=channels, do_3D=False,
                                                  resample=resample,stitch_threshold=stitch_threshold,
                                                  flow_threshold=flow_threshold)
                output_path = os.path.join(output_folder,f"one_model_xy_stitch_{stitch_threshold}_flow_{flow_threshold}", f'{filename_no_extension}.{fileext_out}')
                if not os.path.exists(os.path.dirname(output_path)):
                    os.makedirs(os.path.dirname(output_path))
                itk.imwrite(itk.GetImageFromArray(masks), output_path, compression=True)

    if run_two_models_ortho:
        model2 = models.CellposeModel(pretrained_model=path_model, pretrained_model_ortho=path_second_model, gpu=True)
        try:
            masks, flows, styles = model2.eval(img, diameter=diameter, channels=channels, do_3D=True, resample=resample,
                                              flow3D_smooth=0)
        except Exception as e:
            logging.error(f"Error in {filename_no_extension}: {e}")
            masks, flows, styles = model2.eval(img, diameter=diameter, channels=channels, do_3D=True, resample=False,
                                              flow3D_smooth=0)
        output_path = os.path.join(output_folder, f'two_models_ortho', f'{filename_no_extension}.{fileext_out}')
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        itk.imwrite(itk.GetImageFromArray(masks), output_path, compression=True)
