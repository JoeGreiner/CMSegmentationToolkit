import os
from tifffile import imread
from csbdeep.io import save_tiff_imagej_compatible
from csbdeep.models import CARE
import tensorflow as tf
import glob

from CMSegmentationToolkit.src.restoration.care_restoration import predict_care

if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    import tensorflow as tf
    if tf.test.gpu_device_name():
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    else:
        print("Please install GPU version of TF")

    test_folder = 'data/Confocal_Pinhole/test'
    test_folder_clean = os.path.join(test_folder, 'clean')

    condition_folders = ['p1', 'p2', 'p3', 'p4', 'p5', 'p6',
                         'v2_p06', 'v2_p10', 'v2_p20', 'v2_p30', 'v2_p40', 'v2_p50', 'v2_p60']
    axes = 'ZYX'
    model = CARE(config=None, name='confocal_pinhole_v3', basedir='models')

    for condition_folder in condition_folders:
        all_tifs = glob.glob(os.path.join(test_folder, condition_folder, '*.tif'))
        for file_name in all_tifs:
            x = imread(f'%s' % file_name)
            restored = predict_care(x, model)
            out_folder = f'results/restorations/{condition_folder}/'
            if not os.path.exists(out_folder):
                os.makedirs(out_folder)
            file_name = file_name.split('/')[-1]
            save_tiff_imagej_compatible(f'{out_folder}/{model.name}_{file_name}', restored, axes)
