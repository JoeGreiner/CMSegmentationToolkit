import pandas as pd
import xlsxwriter

def write_df_to_xlsx_auto_col_width(df, path_xlsx_out, sheetName='Sheet1', index_label='stackname'):

    writer = pd.ExcelWriter(path_xlsx_out, engine='xlsxwriter')
    df.to_excel(writer, sheet_name=sheetName, index=True, index_label=index_label, freeze_panes=(1, 1))
    worksheet = writer.sheets[sheetName]
    worksheet.autofit()
    writer.close()
