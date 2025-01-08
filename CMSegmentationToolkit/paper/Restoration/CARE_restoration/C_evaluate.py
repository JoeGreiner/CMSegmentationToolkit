from csbdeep.data import PercentileNormalizer
from csbdeep.utils import normalize_minmse
from skimage.io import imread
import os
import glob
from tqdm import tqdm
import logging
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
import pandas as pd
import xlsxwriter

def write_df_to_xlsx_auto_col_width(df, path_xlsx_out, sheetName='Sheet1', index_label='stackname'):

    writer = pd.ExcelWriter(path_xlsx_out, engine='xlsxwriter')
    df.to_excel(writer, sheet_name=sheetName, index=True, index_label=index_label, freeze_panes=(1, 1))
    worksheet = writer.sheets[sheetName]
    worksheet.autofit()
    writer.close()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

    path_to_test_folder = 'data/Confocal_Pinhole/test/clean'
    path_to_predictions = 'results/restorations'

    model_names = ['confocal_pinhole_v3', ]

    # namefilter = '2024_09_16_pinhole_deblurring.lif - 7'
    namefilter = ''

    conditions = ['p1', 'p2', 'p3', 'p4', 'p5', 'p6',
                  'v2_p06', 'v2_p10', 'v2_p20', 'v2_p30', 'v2_p40', 'v2_p50', 'v2_p60']

    # conditions = ['v2_p60']

    df = pd.DataFrame(columns=[
        'model_name',
        'condition', 'stackname',
        'ssim_raw',
        'ssim_pred',
        'mse_raw',
        'mse_pred',
        'ssim_increase',
        'mse_decrease',
        'path_noisy_input',
        'path_clean',
        'path_pred',
    ], index=[])

    for model_name in model_names:
        logging.info(f'Processing model {model_name}')
        for condition in conditions:
            predictions_path_list = glob.glob(
                os.path.join(path_to_predictions, condition, f'{model_name}*{namefilter}*.tif'))
            for prediction_path in predictions_path_list:

                file_name = os.path.basename(prediction_path).replace(f'{model_name}_', '')
                gt_path = os.path.join(path_to_test_folder, file_name)
                if not os.path.exists(gt_path):
                    print(f'GT not found for {prediction_path}, tried {gt_path}')
                    continue
                noisy_input_path = os.path.join(path_to_test_folder, '..', condition, file_name)
                if not os.path.exists(noisy_input_path):
                    print(f'Noisy input not found for {prediction_path}, tried {noisy_input_path}')
                    continue

                df = pd.concat([df,
                                pd.DataFrame({
                    'path_noisy_input': noisy_input_path,
                    'path_clean': gt_path,
                    'path_pred': prediction_path,
                    'model_name': model_name,
                    'condition': condition,
                    'stackname': file_name}, index=[f'{model_name}_{condition}_{file_name}'])
                                ])

            noisy_input_data_list = []
            clean_array_list = []
            predictions_array_list = []
            stackname_list = []
            for index, row in df.iterrows():
                noisy_input_path = row['path_noisy_input']
                clean_path = row['path_clean']
                pred_path = row['path_pred']

                noisy_input_data = imread(noisy_input_path)
                clean_data = imread(clean_path)
                pred_data = imread(pred_path)

                noisy_input_data_list.append(noisy_input_data)
                clean_array_list.append(clean_data)
                predictions_array_list.append(pred_data)

                stackname_list.append(df.index[df.index == index][0])

        N = len(clean_array_list)

        for i in tqdm(range(N), desc='Calculating SSIM'):
            print(f'Processing {stackname_list[i]}')
            z_dim = clean_array_list[i].shape[0]

            # normalization
            a = PercentileNormalizer()
            normalized_raw = a.before(noisy_input_data_list[i], axes='ZYX')
            normalized_pred = a.before(predictions_array_list[i], axes='ZYX')
            normalized_clean = a.before(clean_array_list[i], axes='ZYX')
            data_range = normalized_clean.max() - normalized_clean.min()

            # scaling
            normalized_raw = normalize_minmse(x=normalized_raw, target=normalized_clean)
            normalized_pred = normalize_minmse(x=normalized_pred, target=normalized_clean)

            # calculate ssim
            ssim_raw = ssim(normalized_clean, normalized_raw, data_range=data_range)
            ssim_pred = ssim(normalized_clean, normalized_pred, data_range=data_range)

            # calculate mse
            mse_raw = mse(normalized_clean, normalized_raw)
            mse_pred = mse(normalized_clean, normalized_pred)

            df.at[stackname_list[i], 'ssim_raw'] = ssim_raw
            df.at[stackname_list[i], 'ssim_pred'] = ssim_pred
            df.at[stackname_list[i], 'mse_raw'] = mse_raw
            df.at[stackname_list[i], 'mse_pred'] = mse_pred

            # ratio ssim
            df.at[stackname_list[i], 'ssim_increase'] = ssim_pred / ssim_raw

            # ratio mse
            df.at[stackname_list[i], 'mse_decrease'] = mse_pred / mse_raw

    # calculate to numeric
    df['ssim_raw'] = pd.to_numeric(df['ssim_raw'])
    df['ssim_pred'] = pd.to_numeric(df['ssim_pred'])
    df['mse_raw'] = pd.to_numeric(df['mse_raw'])
    df['mse_pred'] = pd.to_numeric(df['mse_pred'])
    df['ssim_increase'] = pd.to_numeric(df['ssim_increase'])
    df['mse_decrease'] = pd.to_numeric(df['mse_decrease'])

    # df subset with numeric values
    df_subset = df[
        ['ssim_raw', 'ssim_pred', 'mse_raw', 'mse_pred', 'ssim_increase', 'mse_decrease', 'model_name', 'condition']]

    # calculate average for each method
    df_grouped = df_subset.groupby(['model_name', 'condition']).mean()
    df_super_grouped = df_subset.groupby(['model_name']).mean(numeric_only=True)

    if not os.path.exists('results'):
        os.makedirs('results')

    write_df_to_xlsx_auto_col_width(df, 'results/metrics.xlsx', sheetName='Sheet1', index_label='stackname')
    write_df_to_xlsx_auto_col_width(df_super_grouped, 'results/metrics_grouped.xlsx', sheetName='Sheet1',
                                    index_label='stackname')
    write_df_to_xlsx_auto_col_width(df_grouped, 'results/metrics_super_grouped.xlsx', sheetName='Sheet1',
                                    index_label='stackname')