import logging
import os
import wget
import zipfile

def download(link, outpath, overwrite=False):
    if not os.path.exists(outpath) or overwrite:
        logging.info(f'downloading: {link}')
        wget.download(link, out=outpath)
        logging.info('done')
    else:
        logging.info(f'path {outpath} already exists, not downloading')

def unzip_file(path_to_zip_file: str, path_to_extract_directory: str, delete_zip_afterwards: bool = False):
    with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
        zip_ref.extractall(path_to_extract_directory)
    if delete_zip_afterwards:
        os.remove(path_to_zip_file)

def download_care_model(path_to_download_directory: str, overwrite: bool = False):
    logging.info(f'Downloading model to {path_to_download_directory}')
    if not os.path.exists(path_to_download_directory):
        logging.info(f'Creating directory {path_to_download_directory}')
        os.makedirs(path_to_download_directory)
    url_models = 'https://www.iekm.uniklinik-freiburg.de/storage/s/BbAP7tL6BkknQMJ/download'
    download(url_models, os.path.join(path_to_download_directory, 'models.zip'), overwrite)
    unzip_file(os.path.join(path_to_download_directory, 'models.zip'), path_to_download_directory)

    return os.path.join(path_to_download_directory, 'CARE')

def download_nnu_model(path_to_download_directory: str, overwrite: bool = False):
    logging.info(f'Downloading nnu models to {path_to_download_directory}')
    if not os.path.exists(path_to_download_directory):
        logging.info(f'Creating directory {path_to_download_directory}')
        os.makedirs(path_to_download_directory)
    url_models = 'https://www.iekm.uniklinik-freiburg.de/storage/s/mBYCCLDEW8xjTPj/download'
    download(url_models, os.path.join(path_to_download_directory, 'nnu_models.zip'), overwrite)
    unzip_file(os.path.join(path_to_download_directory, 'nnu_models.zip'), path_to_download_directory)
    return os.path.join(path_to_download_directory, 'nnu_models')


def download_testfiles(path_to_download_directory: str, overwrite: bool = False):
    logging.info(f'Downloading testfiles to {path_to_download_directory}')
    if not os.path.exists(path_to_download_directory):
        logging.info(f'Creating directory {path_to_download_directory}')
        os.makedirs(path_to_download_directory)
    url_testfiles = 'https://www.iekm.uniklinik-freiburg.de/storage/s/RyoWaiBQsA4AaMt/download'
    download(url_testfiles, os.path.join(path_to_download_directory, 'test_data.zip'), overwrite)
    unzip_file(os.path.join(path_to_download_directory, 'test_data.zip'), path_to_download_directory)

    return os.path.join(path_to_download_directory, 'test_data')

def download_testfile_restoration(path_to_download_directory: str, overwrite: bool = False):
    logging.info(f'Downloading testfiles to {path_to_download_directory}')
    if not os.path.exists(path_to_download_directory):
        logging.info(f'Creating directory {path_to_download_directory}')
        os.makedirs(path_to_download_directory)
    url_testfiles = 'https://www.iekm.uniklinik-freiburg.de/storage/s/LH53jTwi9wDgYnB/download'
    download(url_testfiles, os.path.join(path_to_download_directory, 'test_data.tif'), overwrite)
    return os.path.join(path_to_download_directory, 'test_data.tif')

def download_testfile_segmentation(path_to_download_directory: str, overwrite: bool = False):
    logging.info(f'Downloading testfiles to {path_to_download_directory}')
    if not os.path.exists(path_to_download_directory):
        logging.info(f'Creating directory {path_to_download_directory}')
        os.makedirs(path_to_download_directory)
    url_testfiles = 'https://www.iekm.uniklinik-freiburg.de/storage/s/oN6krXoPezZaFye/download'
    download(url_testfiles, os.path.join(path_to_download_directory, 'test_data.nrrd'), overwrite)
    return os.path.join(path_to_download_directory, 'test_data.nrrd')

def download_testfile_morph_analysis(path_to_download_directory: str, overwrite: bool = False):
    logging.info(f'Downloading testfiles to {path_to_download_directory}')
    if not os.path.exists(path_to_download_directory):
        logging.info(f'Creating directory {path_to_download_directory}')
        os.makedirs(path_to_download_directory)
    url_testfiles = 'https://www.iekm.uniklinik-freiburg.de/storage/s/PQEST9E99SWMA3w/download'
    download(url_testfiles, os.path.join(path_to_download_directory, 'test_data_morph.nrrd'), overwrite)
    return os.path.join(path_to_download_directory, 'test_data_morph.nrrd')

def download_testfile_TATS_analysis(path_to_download_directory: str, overwrite: bool = False):
    logging.info(f'Downloading testfiles to {path_to_download_directory}')
    if not os.path.exists(path_to_download_directory):
        logging.info(f'Creating directory {path_to_download_directory}')
        os.makedirs(path_to_download_directory)
    url_testfile_1 = 'https://www.iekm.uniklinik-freiburg.de/storage/s/KnEJGFXNxT3nLZn/download'
    download(url_testfile_1, os.path.join(path_to_download_directory, 'test_seg.nrrd'), overwrite)
    url_testfile_2 = 'https://www.iekm.uniklinik-freiburg.de/storage/s/aTnTgKi7aRoBKFj/download'
    download(url_testfile_2, os.path.join(path_to_download_directory, 'test_wga.nrrd'), overwrite)
    return {
        'segmentation': os.path.join(path_to_download_directory, 'test_seg.nrrd'),
        'wga': os.path.join(path_to_download_directory, 'test_wga.nrrd')
    }


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    # path_to_model = download_care_model('models', overwrite=True)
    # path_to_nnu_model = download_nnu_model('models_nnu', overwrite=True)
    # path_to_testfiles = download_testfile_restoration('test_data', overwrite=True)
    path_to_testfiles = download_testfile_segmentation('test_data', overwrite=True)
    # path_to_testfiles = download_testfiles('test_data', overwrite=False)

    # logging.info(f'Model downloaded to {path_to_model}')
    # logging.info(f'NNU Model downloaded to {path_to_nnu_model}')
    # logging.info(f'Testfiles downloaded to {path_to_testfiles}')
    logging.info('Finished downloading model and testfiles')
