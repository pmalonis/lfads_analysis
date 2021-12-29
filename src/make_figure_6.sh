set -e;
python labeled_example_correction.py; #6a
python labeled_example_traj_co.py; #6b
python controller_average_all_directions_corrections.py #6c
python snr_corrections.py; #6d
cd final_figs;
python hand_v_shoulder_evaluation_correction.py; #6e
python hand_v_shoulder_normalized_corrections.py; #6f
python hand_v_shoulder_magnitude_corrections.py; #6g