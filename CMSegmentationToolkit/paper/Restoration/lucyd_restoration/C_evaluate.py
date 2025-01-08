from csbdeep.data import PercentileNormalizer
from csbdeep.utils import normalize_minmse
from skimage.io import imread
import os
import glob
import pandas as pd
from tqdm import tqdm
import logging
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse

from CMSegmentationToolkit.paper.Restoration.lucyd_restoration.src.write_xlsx import write_df_to_xlsx_auto_col_width

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

    path_to_test_folder = 'predictions/'

    model_names = 'lucyd_norm'

    namefilter = ''

    df = pd.DataFrame()

    all_gts = glob.glob(os.path.join(path_to_test_folder,  '*clear.tif'))
    for gt in tqdm(all_gts):
        matching_blurry = gt.replace('clear', 'blurry')
        matching_pred = gt.replace('_clear', '')
        assert os.path.exists(matching_blurry), f'Could not find {matching_blurry}'
        assert os.path.exists(matching_pred), f'Could not find {matching_pred}'
        stackname = os.path.basename(gt).replace('_clear.tif', '')

        blurry = imread(matching_blurry)
        clear = imread(gt)
        pred = imread(matching_pred)

        # normalise
        a = PercentileNormalizer()
        normalized_raw = blurry
        normalized_pred = a.before(pred, axes='ZYX')
        normalized_clean = clear
        data_range = normalized_clean.max() - normalized_clean.min()

        # scaling
        normalized_raw = normalize_minmse(x=normalized_raw, target=normalized_clean)
        normalized_pred =  normalize_minmse(x=normalized_pred, target=normalized_clean)

       # calculate ssim
        ssim_raw = ssim(normalized_clean, normalized_raw, data_range=data_range)
        ssim_pred = ssim(normalized_clean, normalized_pred, data_range=data_range)

        # calculate mse
        mse_raw = mse(normalized_clean, normalized_raw)
        mse_pred = mse(normalized_clean, normalized_pred)

        # ratio ssim
        ssim_increase = ssim_pred / ssim_raw

        # ratio mse
        mse_decrease = mse_pred / mse_raw

        # condition split at -
        condition = stackname.split('-')[0]

        df = pd.concat([df, pd.DataFrame({
            'path_noisy_input': matching_blurry,
            'path_clean': gt,
            'path_pred': matching_pred,
            'model_name': model_names,
            'condition': condition,
            'stackname': stackname,
            'ssim_raw': ssim_raw,
            'ssim_pred': ssim_pred,
            'mse_raw': mse_raw,
            'mse_pred': mse_pred,
            'ssim_increase': ssim_increase,
            'mse_decrease': mse_decrease,
        }, index=[f'{model_names}_all_{stackname}'])])


    # df subset with numeric values
    df_subset = df[['ssim_raw', 'ssim_pred', 'mse_raw', 'mse_pred', 'ssim_increase', 'mse_decrease', 'model_name', 'condition']]

    # calculate average for each method
    df_grouped = df_subset.groupby(['model_name', 'condition']).mean()

    # drop condition from subset
    df_subset2 = df_subset.drop(columns=['condition'])
    df_super_grouped = df_subset2.groupby(['model_name']).mean()


    if not os.path.exists('results'):
        os.makedirs('results')

    write_df_to_xlsx_auto_col_width(df, 'results/lucyd_metrics.xlsx', sheetName='Sheet1', index_label='stackname')
    write_df_to_xlsx_auto_col_width(df_grouped, 'results/lucyd_metrics_super_grouped.xlsx', sheetName='Sheet1',
                                    index_label='stackname')
    write_df_to_xlsx_auto_col_width(df_super_grouped, 'results/lucyd_metrics_grouped.xlsx', sheetName='Sheet1',
                                    index_label='stackname')
