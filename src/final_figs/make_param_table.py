import pandas as pd
import yaml

param_dict = yaml.safe_load(open('../../lfads_common_parameters.yml'))
param_dict = {k:v for k,v in param_dict.items() if 'log' not in k and 'name' not in k and 'stem' not in k}
param_dict.pop('device')
param_dict.pop('ar_prior_dist')
param_dict.pop('kind')
param_dict.pop('ps_nexamples_to_process')
param_series = pd.Series(param_dict)
column_names = pd.Series({'Parameter Name':'Parameter Value'})  
param_series = pd.concat([column_names, param_series])
param_series.to_csv('../../figures/final_figures/numbered/Table_1.csv')