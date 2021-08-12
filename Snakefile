import os
from glob import glob
import yaml
import subprocess as sp
import sys
sys.path.insert(0, 'src')
from get_lfads_filepaths import write_filepaths

configfile: "config.yml"

locations_path = 'lfads_file_locations.yml'#os.path.join(os.path.dirname(__file__), 'lfads_file_locations.yml')
if not os.path.exists(locations_path):
    write_filepaths(locations_path)

dataset_info = yaml.safe_load(open(locations_path, 'r'))

from select_lfads_model import metric_dict

RAW_DIR = "data/raw/"
INTERMEDIATE_DIR = "data/intermediate/"
MODEL_OUTPUT_DIR = "data/model_output/"
PEAK_DIR = "data/peaks/"
TRIAL_TYPES = ["all"]#["train", "valid", "all"]
SRC_DIR = "src/"
PYTHON_SCRIPTS = glob(SRC_DIR + "*.py")
NOTEBOOKS = glob("notebooks/*.ipynb")
PACKAGES = glob(os.environ["CONDA_PREFIX"] + "/conda-meta/*")
TRIAL_SETS = ["train", "valid"]
CONTROLLER_METRICS = list(metric_dict.keys())

# if sp.check_output(["git", "status", "-s"]):
#     to_run = input("There are uncommitted changes. Run anyway? (y/n):")
#     if to_run.lower() == 'n':
#         sys.exit()

# if open('.git/HEAD').read()[:4]=='ref:': #determines if in detached head state
#     GIT_HEAD = ".git/" + yaml.safe_load(open('.git/HEAD'))['ref']
# else:
#     GIT_HEAD = ".git/HEAD"

# rule git_commit:
#     input:
#         NOTEBOOKS,
#         PYTHON_SCRIPTS,
#         "Snakefile",
#         "config.yml",
#         "environment.yml"
#     output:
#         GIT_HEAD
#     run:
#         shell("git add {input}")
#         shell("git commit")

# rule export_environment:
#     input:
#         PACKAGES
#     output:
#         "environment.yml"
#     shell:
#         "conda env export > {output}"

def expand_filename(format_str):
    '''expands string based on parameter hashes for each dataset in the config file. also expands valid/train'''
    
    filenames = [format_str.replace("{dataset}",d).replace("{param}",p).replace("{trial_type}",t)
                for d in dataset_info.keys()
                for p in dataset_info[d]["params"]
                for t in TRIAL_TYPES]

    filenames = list(set(filenames)) #removing duplicates

    return filenames

rule download_all:
    input:
        expand_filename(RAW_DIR + "{dataset}.mat"),
        expand_filename(MODEL_OUTPUT_DIR + "{dataset}_{param}_inputInfo.mat"),
        expand_filename(MODEL_OUTPUT_DIR + "{dataset}_{param}_{trial_type}.h5")

# rule download_all:
#     output:
#         expand_filename(RAW_DIR + "{dataset}.mat"),
#         expand_filename(MODEL_OUTPUT_DIR + "{dataset}_{param}_inputInfo.mat"),
#         expand_filename(MODEL_OUTPUT_DIR + "{dataset}_{param}_{trial_type}.h5")

#     run:
#         download_list = []
#         output_name_list = []
#         for dataset in dataset_info.keys():
#             download_list.append(dataset_info[dataset]["raw"])
#             output_name_list.append(RAW_DIR + "{dataset}.mat".format(dataset=dataset))
#             for param in dataset_info[dataset]["params"].keys():
#                 download_list.append(dataset_info[dataset]["params"][param]['train'])
#                 output_name_list.append(MODEL_OUTPUT_DIR + "{dataset}_{param}_{trial_type}.mat".format(dataset=dataset, 
#                                                                                     param=param, trial_type="train"))
#                 download_list.append(dataset_info[dataset]["params"][param]['valid'])
#                 output_name_list.append(MODEL_OUTPUT_DIR + "{dataset}_{param}_{trial_type}.mat".format(dataset=dataset, 
#                                                                                     param=param, trial_type="valid"))
#                 download_list.append(dataset_info[dataset]["params"][param]["inputInfo"])
#                 output_name_list.append(MODEL_OUTPUT_DIR + "{dataset}_{param}_inputInfo.mat".format(dataset=dataset, param=param))
                
#         sp.run(['scp', '-T'] + download_list + [MODEL_OUTPUT_DIR])
#         for download, output_file in zip(download_list, output_name_list):
#             os.replace(MODEL_OUTPUT_DIR + os.path.basename(download), output_file)
rule file_locations:
    input:
        "config.yml",
        "src/get_lfads_filepaths.py"

    output:
        "lfads_file_locations.yml"

    script:
        "src/get_lfads_filepaths.py"
        
# rule plot_controller_metric:
#     input:
#         "src/plot_all_controller_metrics.py",
#         "src/plot_controller_metric.py",
#         "lfads_file_locations.yml",
#         expand_filename(MODEL_OUTPUT_DIR + "{dataset}_{param}_inputInfo.mat"),
#         expand_filename(MODEL_OUTPUT_DIR + "{dataset}_{param}_{trial_type}.h5"),
#         expand_filename(INTERMEDIATE_DIR + "{dataset}.p")
#     output:
#         "figures/{metric}.png"

#     script:
#         "src/plot_controller_metric.py"

rule select_model:
    input:
        "src/select_lfads_model.py",
        "lfads_file_locations.yml",
        "config.yml",
        inputInfo_files = lambda wildcards: [(MODEL_OUTPUT_DIR + 
        "{dataset}_{param}_inputInfo.mat").format(dataset=wildcards.dataset, param=p) 
        for p in dataset_info[wildcards.dataset]["params"]],
        
        lfads_files = lambda wildcards: [(MODEL_OUTPUT_DIR + 
        "{dataset}_{param}_all.h5").format(dataset=wildcards.dataset, param=p) 
        for p in dataset_info[wildcards.dataset]["params"]],       
        
        data_file = INTERMEDIATE_DIR + "{dataset}.p",
        
    output:
        PEAK_DIR + "{dataset}_selected_param_%s.txt"%config['selection_metric']

    script:
        "src/select_lfads_model.py"

rule auc_window_fig:
    input:
        # firstmoves = expand_filename(PEAK_DIR + "{dataset}_firstmove_all.p")
        # corrections = expand_filename(PEAK_DIR + "{dataset}_corrections_all.p")
        # maxima = expand_filename(PEAK_DIR + "{dataset}_maxima_all.p")
        # data = 
        # co = 
        # input_info =
        "src/snr.py"
    output:
        firstmove_corrections = "figures/final_figures/firstmove_corrections_auc.svg",
        maxima = "figures/final_figures/maxima_auc.svg"
    script:
        "src/snr.py"


rule target_split:
    input:
        INTERMEDIATE_DIR + "{dataset}.p",
        "config.yml"

    output:
        train_data = PEAK_DIR + "{dataset}_targets_train.p",
        test_data = PEAK_DIR + "{dataset}_targets_test.p",
        all_data = PEAK_DIR + "{dataset}_targets_all.p"

    script:
        "src/target_train-test_split.py"

rule random_split:
    input:
        INTERMEDIATE_DIR + "{dataset}.p",
        "config.yml"

    output:
        train_data = PEAK_DIR + "{dataset}_random_train.p",
        test_data = PEAK_DIR + "{dataset}_random_test.p"

    script:
        "src/random_train-test_split.py"

rule maxima_split:
    input:
        INTERMEDIATE_DIR + "{dataset}.p",
        "src/maxima_train-test_split.py",
        "config.yml"

    output:
        train_data = PEAK_DIR + "{dataset}_maxima_train.p",
        test_data = PEAK_DIR + "{dataset}_maxima_test.p",
        all_data = PEAK_DIR + "{dataset}_maxima_all.p"
    script:
        "src/maxima_train-test_split.py"

rule firstmove_split:
    input:
        INTERMEDIATE_DIR + "{dataset}.p",
        "src/firstmove_train-test_split.py",
        "config.yml"

    output:
        train_data = PEAK_DIR + "{dataset}_firstmove_train.p",
        test_data = PEAK_DIR + "{dataset}_firstmove_test.p",
        all_data = PEAK_DIR + "{dataset}_firstmove_all.p"

    script:
        "src/firstmove_train-test_split.py"

rule correction_split:
    input:
        INTERMEDIATE_DIR + "{dataset}.p",
        "src/correction_train-test_split.py",
        "config.yml"

    output:
        train_data = PEAK_DIR + "{dataset}_corrections_train.p",
        test_data = PEAK_DIR + "{dataset}_corrections_test.p",
        all_data = PEAK_DIR + "{dataset}_corrections_all.p"

    script:
        "src/correction_train-test_split.py"

rule optimize_target_prediction:
    input:
        "config.yml",
        "src/optimize_target_prediction.py",
        selection_files = expand_filename(PEAK_DIR + 
        "{dataset}_selected_param_%s.txt"%config['selection_metric']),
        targets_files = expand_filename(PEAK_DIR + 
        "{dataset}_{event_type}_train.p")
    output:
        PEAK_DIR + "params_search_{event_type}.csv"
    script:
        "src/optimize_target_prediction.py"

rule download_model:
    params:
         source = lambda wildcards: dataset_info[wildcards.dataset]["params"][wildcards.param][wildcards.trial_set]
    output:
        MODEL_OUTPUT_DIR + "{dataset}_{param}_{trial_set}.h5"
    wildcard_constraints:
        trial_set="train|valid"
    run:
        going = True
        while going:
            command = ["scp", "-T", params.source, output[0]]
            proc = sp.Popen(command, stdout=sp.PIPE, stderr=sp.PIPE)
            err = proc.stderr.read().decode()
            if "Connection closed" in err and "port 22" in err:
                continue
            else:
                going=False

rule download_raw:
    params:
        source = lambda wildcards: dataset_info[wildcards.dataset]["raw"]
    output:
        #constrain wildcard to not contain forward slash (doesn't represent file in subdirectory)
        RAW_DIR + "{dataset}.mat"
    run:
        going = True
        while going:
            command = ["scp", "-T", params.source, output[0]]
            proc = sp.Popen(command, stdout=sp.PIPE, stderr=sp.PIPE)
            err = proc.stderr.read().decode()
            if "Connection closed" in err and "port 22" in err:
                continue
            else:
                going=False
                
rule download_center_out:
    params:
        source = lambda wildcards: config["center_out"][wildcards.dataset]
    output:
        RAW_DIR + "center_out/{dataset}.mat"
    shell:
        "scp -T {params.source} {output}"

rule download_inputInfo:
    params:
        source = lambda wildcards: dataset_info[wildcards.dataset]["params"][wildcards.param]["inputInfo"]
    output:
        MODEL_OUTPUT_DIR + "{dataset}_{param}_inputInfo.mat"
    run:
        going = True
        while going:
            command = ["scp", "-T", params.source, output[0]]
            proc = sp.Popen(command, stdout=sp.PIPE, stderr=sp.PIPE)
            err = proc.stderr.read().decode()
            if "Connection closed" in err and "port 22" in err:
                continue
            else:
                going=False

rule download_dataset_inputInfo: #not specific to specific parameter set, just to get trial_len
    params:
        source = lambda wildcards: next(iter(dataset_info[wildcards.dataset]["params"].values()))["inputInfo"]
    wildcard_constraints:
        dataset='|'.join(list(dataset_info.keys()))
    output:
        MODEL_OUTPUT_DIR + "{dataset}_inputInfo.mat"
    run:
        going = True
        while going:
            command = ["scp", "-T", params.source, output[0]]
            proc = sp.Popen(command, stdout=sp.PIPE, stderr=sp.PIPE)
            err = proc.stderr.read().decode()
            if "Connection closed" in err and "port 22" in err:
                continue
            else:
                going=False

rule preprocess_center_out:
    input:
        RAW_DIR + "center_out/{dataset}.mat",
        "src/create_xarray.py"
    output:
        INTERMEDIATE_DIR + "center_out/{dataset}.nc"
    script:
        "src/create_xarray.py"

rule convert_pandas:
    input:
        RAW_DIR + "{dataset}.mat",
        MODEL_OUTPUT_DIR + "{dataset}_inputInfo.mat",
        "src/convert_to_pandas.py"
    wildcard_constraints:
        dataset='|'.join(list(dataset_info.keys()))
    output:
        INTERMEDIATE_DIR + "{dataset}.p"
    script:
        "src/convert_to_pandas.py"

rule plot_inputs:
    input:
        INTERMEDIATE_DIR + "{dataset}.p",
        MODEL_OUTPUT_DIR + "{dataset}_{param}_{trial_type}.h5",
        MODEL_OUTPUT_DIR + "{dataset}_{param}_inputInfo.mat",
        "src/plot_inputs.py"
    output:
        "figures/input_timing_plots/{dataset}_param_{param}_{trial_type}.pdf"
    script:
        "src/plot_inputs.py"

rule plot_all_inputs:
    input:
        expand_filename("figures/input_timing_plots/{dataset}_param_{param}_{trial_type}.pdf")

rule decode_lfads:
    input:
        INTERMEDIATE_DIR + "{dataset}.p",
        MODEL_OUTPUT_DIR + "{dataset}_{param}_{trial_type}.h5",
        MODEL_OUTPUT_DIR + "{dataset}_{param}_inputInfo.mat",
        "src/decode_lfads.py"
    output:
        "figures/decode_from_lfads_output/{dataset}_{param}_{trial_type}_decode_from_output.pdf",
        "figures/decode_from_lfads_output/{dataset}_{param}_{trial_type}_decode_from_factors.pdf"
    script:
        "src/decode_lfads.py"

rule decode_all:
    input:
        expand_filename("figures/decode_from_lfads_output/{dataset}_{param}_{trial_type}_decode_from_output.pdf")

rule input_analysis:
    input:
        INTERMEDIATE_DIR + "{dataset}.p",
        MODEL_OUTPUT_DIR + "{dataset}_{param}_{trial_type}.h5",
        MODEL_OUTPUT_DIR + "{dataset}_{param}_inputInfo.mat",
        SRC_DIR + "process_inputs.py",
    log:
        notebook = "notebooks/processed/{dataset}_{param}_{trial_type}_integral_analysis.ipynb"

rule rate_pcs_search:
    input:
        PEAK_DIR + 'params_search_{event_type}.csv'
    output:
        PEAK_DIR + 'rate_pcs_search_{event_type}.csv',
    script:
        "src/rate_pcs_search.py"

rule target_prediction_evaluation_controller:
    input:
        PEAK_DIR + 'params_search_{event_type}.csv'
    params:
        reference = ['hand', 'shoulder'],
        use_rates = [False] #get maximum of reference only for controller prediction, not rate prediction
    output:
        PEAK_DIR + 'controller_best_models_{event_type}.csv'
    script:
        "src/model_evaluation.py"

rule rate_pcs_target_prediction_evaluation:
    input:
        PEAK_DIR + 'rate_pcs_search_{event_type}.csv',
        "src/model_evaluation.py"
    params:
        filter_co = [True]
    output:
        PEAK_DIR + 'rate_best_models_{event_type}.csv'
    script:
        "src/model_evaluation.py"

rule hand_v_shoulder_fig:
    input:
        PEAK_DIR + 'controller_best_models_firstmove.csv',
        PEAK_DIR + 'controller_best_models_corrections.csv',
        "src/final_figs/hand_v_shoulder_evaluation.py"
    output:
        'figures/final_figures/hand_v_shoulder_initial.png'
    script:
        "src/final_figs/hand_v_shoulder_evaluation.py"

rule rate_pcs_fig:
    input:
        PEAK_DIR + 'controller_best_models_firstmove.csv',
        PEAK_DIR + 'controller_best_models_corrections.csv',
        "src/final_figs/rate_decoding_vs_co.py"
    output:
        "figures/final_figs/rate_decoding_firstmove.svg",
        "figures/final_figs/rate_decoding_corretive.svg"
    script:
        "src/final_figs/rate_decoding_vs_co.py"

rule window_search_fig:
    input:
        PEAK_DIR + "params_search_firstmove.csv",
        PEAK_DIR + "params_search_corrections.csv",
        "src/final_figs/plot_window_performance.py"
    output:
        "figures/final_figures/window_performance.svg"
    script:
        "src/final_figs/plot_window_performance.py"

rule hand_time_search_fig:
    input:
        PEAK_DIR + "params_search_firstmove.csv",
        PEAK_DIR + "params_search_corrections.csv",
        "src/final_figs/plot_hand_time_performance.py"
    output:
        "figures/final_figures/hand_time.svg"
    script:
        "src/plot_hand_time_performance.py"

rule random_event_performance_fig:
    input:
        PEAK_DIR + "controller_best_models_random.csv",
        PEAK_DIR + "rate_best_models_random.csv",
        "src/final_figs/random_event_performance.py"
    output:
        "figures/final_figures/random_events.svg"
    script:
        "src/final_figs/random_event_performance.py"

rule maxima_event_performance_fig:
    input:
        PEAK_DIR + "controller_best_models_maxima.csv",
        PEAK_DIR + "rate_best_models_maxima.csv",
        "src/final_figs/maxima_event_performance.py"
    output:
        "figures/final_figures/speed_maxima.svg"
    script:
        "src/final_figs/maxima_event_performance.py"

rule all_figs:
    input:
        #"figures/final_figures/random_events.svg",
        #"figures/final_figures/hand_time.svg",
        "figures/final_figures/window_performance.svg",
        "figures/final_figs/rate_decoding_firstmove.svg",
        "figures/final_figs/rate_decoding_corretive.svg",
        'figures/final_figures/hand_v_shoulder.svg'

 #   notebook:
 #       "notebooks/integral_analysis.ipynb"

# rule full_input_analysis:
#     input:
#         INTERMEDIATE_DIR + "{dataset}.p",
#         MODEL_OUTPUT_DIR + "{dataset}_{param}_{trial_type}.h5",
#         MODEL_OUTPUT_DIR + "{dataset}_{param}_inputInfo.mat",
#         SRC_DIR + "process_inputs.py",
#     log:
#         notebook = "notebooks/processed/{dataset}_{param}_{trial_type}_input_analysis.ipynb"
#     notebook:
#         "notebooks/input_analysis.ipynb"

rule peak_analysis:
    input:
        INTERMEDIATE_DIR + "{dataset}.p",
        MODEL_OUTPUT_DIR + "{dataset}_{param}_{trial_type}.h5",
        MODEL_OUTPUT_DIR + "{dataset}_{param}_inputInfo.mat",
        SRC_DIR + "process_inputs.py",
    log:
        notebook = "notebooks/processed/{dataset}_{param}_{trial_type}_peak_analysis.ipynb"
#    notebook:
#        "notebooks/peak_analysis.ipynb"

rule simulated_data_notebook:
    input:
        SRC_DIR + "lds_regression.py",
        SRC_DIR + "evaluate_all_datasets.py",
        SRC_DIR + "glds.py"
    log:
        notebook = "notebooks/processed/simulated_data.ipynb"
#    notebook:
#        "notebooks/simulated_data.ipynb"
        
rule notebook_to_html:
    input:
        "notebooks/processed/{filename}.ipynb"
    output:
        "figures/{filename}.html"
    run:
        shell("jupyter nbconvert --to html %s --output-dir . --output %s"%(input[0], output[0]))
        with open(output[0], 'r') as read_file:
            html_text = read_file.read()
        commit = sp.check_output(['git', 'rev-parse', 'HEAD']).strip()
        html_text = "<!-- commit: %s-->\n"%commit + html_text
        with open(output[0], 'w') as write_file:
            write_file.write(html_text)        

rule movies_with_inputs:
    input:
        MODEL_OUTPUT_DIR + "{dataset}_{param}_{trial_type}.h5",
        RAW_DIR + "{dataset}.mat",
        MODEL_OUTPUT_DIR + "{dataset}_{param}_inputInfo.mat",
        "src/kinematics_and_input.py"
    output:
        dynamic("figures/kinematics_movies_with_inputs/{dataset}_{param}_{trial_type}/trial_{trial_number}.mp4")
    script:
        "src/kinematics_and_input.py"

def get_movie_filenames(wildcards):
    out_directory = "figures/kinematics_movies_with_inputs/%s_%s_%s/"%(wildcards.dataset, wildcards.param, wildcards.trial_type)
    filenames = glob(out_directory + "trial_*.mp4")
    if len(filenames) == 0:
        filenames.append(out_directory + "trial_000.mp4")

    return filenames

rule concatenate_movies:
    input:
        get_movie_filenames 
    output:
        "figures/kinematics_movies_with_inputs/concatenated_{dataset}_{param}_{trial_type}.mp4"
    run:
        out_directory = os.path.dirname(output[0])        
        file_list = out_directory + "/temp_input_kinematics_movie_file_list.txt"
        filenames = []
        for ip in input:
            directory, base = os.path.split(ip)
            rel_dir = os.path.basename(directory)
            filenames.append(rel_dir + '/' + base)

        with open(file_list, 'w') as f:
            f.writelines(["file \'%s\'\n"%filename for filename in filenames])
        shell("ffmpeg -f concat -safe 0 -i %s -c copy %s"%(file_list, output[0]))
        shell("rm %s"%(file_list))

rule process_inputs:
    input:
        INTERMEDIATE_DIR + "{dataset}.p",
        MODEL_OUTPUT_DIR + "{dataset}_{param}_{trial_type}.h5",
        MODEL_OUTPUT_DIR + "{dataset}_{param}_inputInfo.mat",
        "src/process_inputs.py"
    output:
        "data/processed_inputs/{dataset}_{param}_{trial_type}.p"
    script:
        "src/process_inputs.py"

rule process_all_inputs:
    input:
        expand_filename(INTERMEDIATE_DIR + "processed_inputs/processed_inputs_{dataset}_{param}_{trial_type}.p")

rule combine_trials:
    input:
        MODEL_OUTPUT_DIR + "{dataset}_{param}_train.h5",
        MODEL_OUTPUT_DIR + "{dataset}_{param}_valid.h5",
        MODEL_OUTPUT_DIR + "{dataset}_{param}_inputInfo.mat",
        "src/combine_lfads_output.py"
    output:
        MODEL_OUTPUT_DIR + "{dataset}_{param}_all.h5"
    script:
        "src/combine_lfads_output.py"

rule random_forest_predict:
    input:
        "data/processed_inputs/{dataset}_{param}_{trial_type}.p",
        "src/predict_targets_control.py"
    output:
        "figures/target_prediction/{dataset}_{param}_{trial_type}.pdf"
    script:
        "src/predict_targets.py"

rule random_forest_predict_control:
    input:
        "data/processed_inputs/{dataset}_{param}_{trial_type}.p",
        "src/predict_targets_control.py"
    output:
        "figures/target_prediction_control/{dataset}_{param}_{trial_type}.pdf"
    script:
        "src/predict_targets_control.py"

rule predict_all:
    input:
        expand_filename("figures/target_prediction/{dataset}_{param}_{trial_type}.pdf")

rule make_all_movies:
    input:
        expand_filename("figures/kinematics_movies_with_inputs/concatenated_{dataset}_{param}_{trial_type}.mp4")

rule split_playback:
    input:
        "data/raw/Playback-NN/P-{dataset}.mat"
    output:
        "data/raw/Playback-NN/split_condition/{dataset}/active.mat",
        "data/raw/Playback-NN/split_condition/{dataset}/vis_pb.mat",
        "data/raw/Playback-NN/split_condition/{dataset}/prop_pb.mat",
        "data/raw/Playback-NN/split_condition/{dataset}/dual_pb.mat"
    script:
        "src/playback_to_lfads.py"

rule all_split_playback:
    input:
        "data/raw/Playback-NN/split_condition/b080723_M1/active.mat",
        "data/raw/Playback-NN/split_condition/b080723_M1/vis_pb.mat",
        "data/raw/Playback-NN/split_condition/b080723_M1/prop_pb.mat",
        "data/raw/Playback-NN/split_condition/b080723_M1/dual_pb.mat"
        
        "data/raw/Playback-NN/split_condition/b080725_M1/active.mat",
        "data/raw/Playback-NN/split_condition/b080725_M1/vis_pb.mat",
        "data/raw/Playback-NN/split_condition/b080725_M1/prop_pb.mat",
        "data/raw/Playback-NN/split_condition/b080725_M1/dual_pb.mat"
        
        "data/raw/Playback-NN/split_condition/b080905_M1/active.mat",
        "data/raw/Playback-NN/split_condition/b080905_M1/vis_pb.mat",
        "data/raw/Playback-NN/split_condition/b080905_M1/prop_pb.mat",
        "data/raw/Playback-NN/split_condition/b080905_M1/dual_pb.mat"

        "data/raw/Playback-NN/split_condition/mk080729_M1m/active.mat",
        "data/raw/Playback-NN/split_condition/mk080729_M1m/vis_pb.mat",
        "data/raw/Playback-NN/split_condition/mk080729_M1m/prop_pb.mat",
        "data/raw/Playback-NN/split_condition/mk080729_M1m/dual_pb.mat"
        
        "data/raw/Playback-NN/split_condition/mk080730_M1m/active.mat",
        "data/raw/Playback-NN/split_condition/mk080730_M1m/vis_pb.mat",
        "data/raw/Playback-NN/split_condition/mk080730_M1m/prop_pb.mat",
        "data/raw/Playback-NN/split_condition/mk080730_M1m/dual_pb.mat"
        
        "data/raw/Playback-NN/split_condition/mk080731_M1m/active.mat",
        "data/raw/Playback-NN/split_condition/mk080731_M1m/vis_pb.mat",
        "data/raw/Playback-NN/split_condition/mk080731_M1m/prop_pb.mat",
        "data/raw/Playback-NN/split_condition/mk080731_M1m/dual_pb.mat"

        "data/raw/Playback-NN/split_condition/mk080828_M1m/active.mat",
        "data/raw/Playback-NN/split_condition/mk080828_M1m/vis_pb.mat",
        "data/raw/Playback-NN/split_condition/mk080828_M1m/prop_pb.mat",
        "data/raw/Playback-NN/split_condition/mk080828_M1m/dual_pb.mat"