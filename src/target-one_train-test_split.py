import pandas as pd
import os
from sklearn.model_selection import train_test_split
import timing_analysis as ta
from importlib import reload
reload(ta)

train_filename = snakemake.output.train_data
test_filename = snakemake.output.test_data
all_filename = snakemake.output.all_data
data_filename = snakemake.input[0]

random_state = 1027
train_test_ratio = 0.2

def split_target_df(df):
    target_df = ta.get_targets(df)
    target_df = target_df.groupby('trial').apply(ta.assign_target_column)
    # taking first target only
    target_df = target_df.groupby('trial').apply(lambda _df: _df.loc[_df.index[0][0]].iloc[:1]) 

    df_train, df_test = train_test_split(target_df, test_size=train_test_ratio, random_state=random_state)
    df_train, df_test = (df_train.sort_index(), df_test.sort_index())
    df_train.to_pickle(train_filename)
    df_test.to_pickle(test_filename)
    target_df.to_pickle(all_filename)
    
    return df_train, df_test

if __name__=='__main__':
    df = pd.read_pickle(data_filename)
    split_target_df(df)