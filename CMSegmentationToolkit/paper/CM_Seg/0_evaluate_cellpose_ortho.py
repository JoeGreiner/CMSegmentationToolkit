import os
import shutil
import pandas as pd
import glob
from CMSegmentationToolkit.src.evaluate import evaluatate_single_image

if __name__ == '__main__':
    file_ext = 'nrrd'
    base_path = '/mnt/work/data/CM_Seg_Paper_24/data_v2/'
    path_xlsx_out = os.path.join(base_path, 'results', 'tables', 'cellpose_ortho_evaluation.xlsx')
    prediction_folder = '/mnt/work/data/CM_Seg_Paper_24/results/cellpose_test_20250316-133926/two_models_ortho'

    all_nrrd_pred = glob.glob(os.path.join(prediction_folder, '*.nrrd'), recursive=True)

    df = pd.DataFrame()
    for path_to_pred in all_nrrd_pred:
        if 'human' in path_to_pred:
            print(f'Skipping {path_to_pred}')
            continue
        stackname = os.path.basename(path_to_pred)
        if '_mc_0.' in stackname:
            stackname = stackname.split('_mc_0.')[0]
        if '_masks_' in stackname:
            stackname = stackname.split('_masks_')[0]
        if '.' in stackname:
            stackname = stackname.split('.')[0]
        path_to_gt = glob.glob(os.path.join(base_path, '*', 'test', 'labels', f'{stackname}*.nrrd'), recursive=True)
        if len(path_to_gt) == 0:
            print(f'Could not find gt for {stackname}')
            continue
        path_to_gt = path_to_gt[0]

        df_single = evaluatate_single_image(path_to_gt, path_to_pred)
        df = pd.concat([df, df_single])

    dirname_output = os.path.dirname(path_xlsx_out)
    if not os.path.exists(dirname_output):
        os.makedirs(dirname_output)

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
