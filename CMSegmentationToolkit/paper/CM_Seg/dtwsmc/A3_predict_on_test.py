import glob

from CMSegmentationToolkit.src.dtws_mc import run_freiburg_mc

if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    output_folder_base = '/mnt/work/data/CM_Seg_Paper_24/results/freiburg_mc/'
    all_test_files = glob.glob('/mnt/work/data/CM_Seg_Paper_24/data_v2/*/test/imgs_norm/*.nrrd', recursive=True)
    for img_path in all_test_files:
        run_freiburg_mc(img_path, output_folder_base, betas=[0.075], betas2=[])
