source activate lfads_analysis
echo Finding initial movements
python new_firstmove_train-test_split.py
echo Finding corrections
python new_correction_train-test_split.py
python pdf_example_correction.py
python pdf_example_traj_co.py
source activate base