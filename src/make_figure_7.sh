set -e;
python controller_average_combined.py; #7a
python direction-averages_correlation.py; #7b
python direction-averages_correlation-rates.py; #7c
python maxima_vs_firstmove_direction-averages_correlation-rates.py; #7d