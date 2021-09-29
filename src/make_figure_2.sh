set -e;
cd final_figs;
python co_dim_decode.py;
cd ..;
python plot_all_controller_metrics.py; #2c #2d
python plot_all_controller_metrics_co_dim.py; #2e
python co_dim_compare_target_decode_separate.py #2f