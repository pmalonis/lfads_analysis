import os
from glob import glob
import yaml
import subprocess as sp
import sys

configfile: "config.yml"

RAW_DIR = "data/raw/"
INTERMEDIATE_DIR = "data/intermediate/"
MODEL_OUTPUT_DIR = "data/model_output/"
TRIAL_TYPES = ["train", "valid", "all"]
SRC_DIR = "src/"
PYTHON_SCRIPTS = glob(SRC_DIR + "*.py")
NOTEBOOKS = glob("notebooks/*.ipynb")
PACKAGES = glob(os.environ["CONDA_PREFIX"] + "/conda-meta/*")
TRIAL_SETS = ["train", "valid"]

if sp.check_output(["git", "status", "-s"]):
    to_run = input("There are uncommitted changes. Run anyway? (y/n):")
    if to_run.lower() == 'n':
        sys.exit()

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
    
    filenames = [format_str.format(dataset=d, trial_type=t, param=p) for d in config["datasets"].keys()
                for p in config["datasets"][d]["params"]
                for t in TRIAL_TYPES]

    return filenames

rule download_all:
    input:
        expand_filename(RAW_DIR + "{dataset}.mat"),
        expand_filename(MODEL_OUTPUT_DIR + "{dataset}_{param}_inputInfo.mat"),
        expand_filename(MODEL_OUTPUT_DIR + "{dataset}_{param}_{trial_type}.h5"),

rule download_model:
    params:
         source = lambda wildcards: config["datasets"][wildcards.dataset]["params"][wildcards.param][wildcards.trial_set]
    output:
        MODEL_OUTPUT_DIR + "{dataset}_{param}_{trial_set}.h5"
    wildcard_constraints:
        trial_set="train|valid"
    shell:
        #"scp -T {config[username]}@{params.source} {output}"
        "scp -T {params.source} {output}"

rule download_raw:
    params:
        source = lambda wildcards: config["datasets"][wildcards.dataset]["raw"]
    output:
        RAW_DIR + "{dataset}.mat"
    shell:
        #"scp -T {config[username]}@{params.source} {output}"
        "scp -T {config[username]}@{params.source} {output}"

rule download_inputInfo:
    params:
         source = lambda wildcards: config["datasets"][wildcards.dataset]["params"][wildcards.param]["inputInfo"]
    output:
        MODEL_OUTPUT_DIR + "{dataset}_{param}_inputInfo.mat"
    shell:
        #"scp -T {config[username]}@{params.source} {output}"
        "scp -T {params.source} {output}"

rule convert_pandas:
    input:
        RAW_DIR + "{dataset}.mat",
        MODEL_OUTPUT_DIR + "{dataset}_inputInfo.mat",
        "src/convert_to_pandas.py"
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
        expand_filename("figures/input_timing_plots/{dataset}_param_{param}_{trial_type}.pdf"),
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
    notebook:
        "notebooks/integral_analysis.ipynb"

rule peak_analysis:
    input:
        INTERMEDIATE_DIR + "{dataset}.p",
        MODEL_OUTPUT_DIR + "{dataset}_{param}_{trial_type}.h5",
        MODEL_OUTPUT_DIR + "{dataset}_{param}_inputInfo.mat",
        SRC_DIR + "process_inputs.py",
    log:
        notebook = "notebooks/processed/{dataset}_{param}_{trial_type}_peak_analysis.ipynb"
    notebook:
        "notebooks/peak_analysis.ipynb"

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
        MODEL_OUTPUT_DIR + "{dataset}_inputInfo.mat",
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
        MODEL_OUTPUT_DIR + "{dataset}_inputInfo.mat",
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
        MODEL_OUTPUT_DIR + "{dataset}_inputInfo.mat",
        "src/combine_lfads_output.py"
    output:
        MODEL_OUTPUT_DIR + "{dataset}_{param}_all.h5"
    script:
        "src/combine_lfads_output.py"

rule random_forest_predict:
    input:
        "data/processed_inputs/rockstar_kuGTbO_all.p"

    output:
        "figures/r^2 Target Prediction.png",
        "figures/Mean distance (mm) Target Prediction.png",
        "figures/Categorical Accuracy (%) Target Prediction.png"

    script:
        "src/predict_targets.py"

rule make_all_movies:
    input:
        expand_filename("figures/kinematics_movies_with_inputs/concatenated_{dataset}_{param}_{trial_type}.mp4")