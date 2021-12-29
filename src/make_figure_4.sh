set -e;
python controller_average_all_directions.py; # 4a
cd final_figs
python peak_timing.py; #4b
python maximum_peak_timing_comparison.py; # 4c
cd ..;
python snr_targets.py #4d