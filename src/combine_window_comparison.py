import pandas as pd

def combine(output, raju_output):
    old = output.query('dataset != "RJ" | use_rates')
    return pd.concat([old, raju_output], ignore_index=True)

event = 'targets-not-one'

output = pd.read_csv('../data/peaks/old_window_comparison_%s.csv'%event)
raju_output = pd.read_csv('../data/peaks/raju_window_comparison_%s.csv'%event)

output_combined = combine(output, raju_output)
output_combined.to_csv('../data/peaks/window_comparison_%s.csv'%event)