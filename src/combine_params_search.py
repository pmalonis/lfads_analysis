import pandas as pd

def combine(output, raju_output):
    old = output.query('dataset != "RJ" | use_rates')
    return pd.concat([old, raju_output], ignore_index=True)

events = ['targets-not-one', 'corrections']
for event in events:
    output = pd.read_csv('../data/peaks/old_raju_params_search_%s.csv'%event)
    raju_output = pd.read_csv('../data/peaks/raju_params_search_%s.csv'%event)

    output_combined = combine(output, raju_output)
    output_combined.to_csv('../data/peaks/params_search_%s.csv'%event)