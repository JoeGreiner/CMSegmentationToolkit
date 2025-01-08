import pandas as pd

if __name__ == '__main__':
    # load the dataframe
    df = pd.read_excel('/home/greinerj/PycharmProjects/CM_Seg/dtwsmc/multicut_tuning_vali_fold_0.xlsx')
    df['avg_merge_split'] = (df['merge_sk'] + df['split_sk']) / 2
    # check all the combinations, average over all stacks, find the one with the best mean_merge_split (lowest)
    # first src: all that doesnt have mc2 value, and then the best mc_value for all stacknames
    if 'mc2_value' in df.columns:
        df_one_mc = df[df['mc2_value'].isna()]
    else:
        df_one_mc = df
    # drop everything but mc_value and stackname
    df_one_mc = df_one_mc.groupby('mc_value').mean(numeric_only=True).reset_index()
    df_one_mc = df_one_mc.sort_values(by='mc_value')
    print(df_one_mc)
    df_one_mc.to_excel('multicut_tuning_vali_fold_0_one_mc.xlsx')

    # second src: all that have mc2 value, and then the best mc_value for all stacknames
    if 'mc2_value' not in df.columns:
        print('No mc2_value in df')
        exit(1)
    df_two_mc = df[~df['mc2_value'].isna()]
    # group by unique combination of mc_value and mc2_value
    df_two_mc = df_two_mc.groupby(['mc_value', 'mc2_value']).mean(numeric_only=True).reset_index()
    df_two_mc = df_two_mc.sort_values(by='are_sk')
    print(df_two_mc)

    # save to xlsx
    df_two_mc.to_excel('multicut_tuning_vali_fold_0_two_mc.xlsx')

