import os
import numpy as np
import pandas as pd
import xlsxwriter
import logging

def create_folder_of_path_if_not_exists(path):
    folder = os.path.dirname(path)
    if not folder:
        return
    create_folder_if_not_exists(folder)

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def test_is_numeric_column():
    series_1 = pd.Series([1, 2, 3, 4, 5])
    assert is_numeric_column(series_1) == True, "Test 1 failed"

    series_2 = pd.Series([1, 2, 3, 4, 5, 'a'])
    assert is_numeric_column(series_2) == False, "Test 2 failed"

    series_3 = pd.Series([1, 2, 3, 4, 5, None])
    assert is_numeric_column(series_3) == True, "Test 3 failed"

    series_4 = pd.Series([1, 2, 3, 4, 5, np.nan])
    assert is_numeric_column(series_4) == True, "Test 4 failed"

    series_5 = pd.Series([1, 2, 3, 4, 5, ''])
    assert is_numeric_column(series_5) == True, "Test 5 failed"

    series_6 = pd.Series(["a"])
    assert is_numeric_column(series_6) == False, "Test 6 failed"


def is_numeric_column(column : pd.Series) -> bool:
    column = column.dropna()
    column = column[column != '']
    numeric_column = pd.to_numeric(column, errors='coerce')
    return numeric_column.notna().all() or column.isna().all()

def export_dataframe_to_excel_autofit(df, filename, sheet_name='Results', color_numeric=True,
                                      verbose=False, backup_if_exists=True):
    if backup_if_exists:
        if os.path.exists(filename):
            logging.info(f'Backing up existing file {filename}')
            datestr = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            # if filename of type posixpath convert to string
            if isinstance(filename, os.PathLike):
                filename = str(filename)
            backup_filename = filename.replace('.xlsx', f'_{datestr}.xlsx')
            backup_folder = os.path.join(os.path.dirname(filename), 'backup')
            create_folder_if_not_exists(backup_folder)
            output_path = os.path.join(backup_folder, os.path.basename(backup_filename))
            os.rename(filename, output_path)

    # create folder if not exists
    create_folder_of_path_if_not_exists(filename)

    with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=True)
        workbook = writer.book
        worksheet = writer.sheets[sheet_name]

        header_format = workbook.add_format({
            'bold': True,
            'bg_color': '#000000',
            'font_color': '#FFFFFF',
            'border': 1,
            'align': 'center',
            'valign': 'vcenter',
        })

        # get dimensions of the multi index, it can be more than one!
        if isinstance(df.index, pd.MultiIndex):
            index_names = df.index.names
            number_index_levels = len(index_names)
            for idx, name in enumerate(index_names):
                worksheet.write(0, idx, name, header_format)
        else:
            number_index_levels = 1
            worksheet.write(0, 0, 'Index', header_format)

        for col_num, value in enumerate(df.columns.values):
            worksheet.write(0, col_num + number_index_levels, value, header_format)

        worksheet.autofit()

        worksheet.freeze_panes(1, 1) # freeze first row and first column

        if color_numeric:
            for col_num, column in enumerate(df.columns, start=1):
                if is_numeric_column(df[column]):
                    excel_column = xlsxwriter.utility.xl_col_to_name(col_num)
                    end_row = len(df)
                    cell_range = f"{excel_column}2:{excel_column}{end_row + 1}"

                    try:
                        min_value = df[column].min()
                        max_value = df[column].max()
                        median_value = df[column].median()
                    except TypeError as e:
                        print(f'Error in column {column}: {e}')
                        continue

                    format_color_scale = {
                        'type': '3_color_scale',
                        'min_type': 'num',
                        'min_value': min_value,
                        'min_color': "#FF0000",
                        'mid_type': 'num',
                        'mid_value': median_value,
                        'mid_color': "#FFFF00",  # Yellow for median
                        'max_type': 'num',
                        'max_value': max_value,
                        'max_color': "#00FF00",
                    }

                    worksheet.conditional_format(cell_range, format_color_scale)
                else:
                    if verbose:
                        print(f'Column {column} is not numeric, skipping color formatting.')



def trim_if_endswidth_filter(stackname, filter):
    if stackname.endswith(filter):
        stackname = stackname[:-len(filter)]
    return stackname

def trim_if_startswith_filter(stackname, filter):
    if stackname.startswith(filter):
        stackname = stackname[len(filter):]
    return stackname

