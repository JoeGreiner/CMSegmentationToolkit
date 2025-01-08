import os
import shutil

import pandas as pd

from CMSegmentationToolkit.src.evaluate import evaluatate_single_image

if __name__ == '__main__':
    file_ext = 'nrrd'
    base_path = '/mnt/work/data/CM_Seg_Paper_24/data_v2/'
    path_xlsx_out = os.path.join(base_path, 'results', 'tables', 'human_evaluation.xlsx')
    prediction_folder = '/mnt/work/data/CM_Seg_Paper_24/results/freiburg_mc_2024_10_02/prediction_combined/'
    import glob
    assert os.path.exists(prediction_folder), f'{prediction_folder} does not exist'

    all_nrrd_pred = glob.glob(os.path.join(prediction_folder, '*', '*pmap_zero.nrrd'), recursive=True)

    df = pd.DataFrame()
    for path_to_pred in all_nrrd_pred:

        stackname = os.path.basename(path_to_pred).replace('_pmap_zero.nrrd', '')
        if '_mc_0.' in stackname:
            stackname = stackname.split('_mc_0.')[0]
        path_to_gt = glob.glob(os.path.join(base_path, '*', 'test', 'labels', f'{stackname}.nrrd'), recursive=True)
        if len(path_to_gt) == 0:
            print(f'Could not find gt for {stackname}')
            continue
        path_to_gt = path_to_gt[0]

        if not 'human' in path_to_gt:
            print(f'Skipping {path_to_pred}')
            continue

        df_single = evaluatate_single_image(path_to_gt, path_to_pred)
        df = pd.concat([df, df_single])

    dirname_output = os.path.dirname(path_xlsx_out)
    if not os.path.exists(dirname_output):
        os.makedirs(dirname_output)

    # if file exists, copy it to a backup file with a timestamp
    backup_folder = os.path.join(dirname_output, 'backup')
    if not os.path.exists(backup_folder):
        os.makedirs(backup_folder)
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    old_filename = os.path.splitext(os.path.basename(path_xlsx_out))[0]
    backup_file = os.path.join(backup_folder, f'{old_filename}_{timestamp}.xlsx')
    if os.path.exists(path_xlsx_out):
        shutil.copy(path_xlsx_out, backup_file)
        print(f'Backup of {path_xlsx_out} saved to {backup_file}')

    with pd.ExcelWriter(path_xlsx_out, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Sheet1')
        worksheet = writer.sheets['Sheet1']
        worksheet.autofit()

    # show mean of adapted rand error, merge, and split error
    # are_sk split_sk merge_sk
    print(f'Mean ARE: {df["are_sk"].mean()}')
    print(f'Mean Split: {df["split_sk"].mean()}')
    print(f'Mean Merge: {df["merge_sk"].mean()}')