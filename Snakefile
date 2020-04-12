import os
from glob import glob
import yaml
import subprocess as sp
import sys

configfile: "config.yml"

RAW_DIR = "data/raw/"
INTERMEDIATE_DIR = "data/intermediate/"
MODEL_OUTPUT_DIR = "data/model_output/"
TRIAL_TYPES = ["train", "valid"]
SRC_DIR = "src/"
PYTHON_SCRIPTS = glob(SRC_DIR + "*.py")
NOTEBOOKS = glob("notebooks/*.ipynb")
PACKAGES = glob(os.environ["CONDA_PREFIX"] + "/conda-meta/*")

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
        expand_filename(MODEL_OUTPUT_DIR + "{dataset}_{param}_{trial_type}.h5")

rule download_model:
    params:
         source = lambda wildcards: config["datasets"][wildcards.dataset]["params"][wildcards.param][wildcards.trial_type]
    output:
        MODEL_OUTPUT_DIR + "{dataset}_{param}_{trial_type}.h5"
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
        "figures/kinematics_movies_with_inputs/{dataset}_{param}_{trial_type}.mp4"
    script:
        "src/kinematics_and_input.py"

# rule input_analysis:
#     input:
#         "data/intermediate/rockstar.p",
#         "data/model_output/rockstar_inputInfo.mat",
#         "data/model_output/rockstar_valid.h5",
#         SRC_DIR + "process_inputs.py",
#         "{input_type}_analysis.ipynb"
#     output:
#         "{input_type}_analysis.html"
#     shell:
#         "jupyter nbconvert --to notebook --inplace --execute {input[4]} && jupyter nbconvert --to html {input[4]}" 