import os
from glob import glob
import yaml

configfile: "config.yml"

RAW_DIR = "data/raw/"
INTERMEDIATE_DIR = "data/intermediate/"
MODEL_OUTPUT_DIR = "data/model_output/"
OUTPUT_TYPES = ["train", "valid"]
SRC_DIR = "src/"
PYTHON_SCRIPTS = glob(SRC_DIR + "*.py")
NOTEBOOKS = glob("notebooks/*.ipynb")
PACKAGES = glob(os.environ["CONDA_PREFIX"] + "/conda-meta/*")

if open('.git/HEAD').read()[:4]=='ref:': #determines if in detached head state
    GIT_HEAD = ".git/" + yaml.safe_load(open('.git/HEAD'))['ref']
else:
    GIT_HEAD = ".git/HEAD"

rule git_commit:
    input:
        NOTEBOOKS,
        PYTHON_SCRIPTS,
        "Snakefile",
        "config.yml",
        "environment.yml"
    output:
        GIT_HEAD
    run:
        shell("git add {input}")
        shell("git commit")

rule export_environment:
    input:
        PACKAGES
    output:
        "environment.yml"
    shell:
        "conda env export > {output}"

rule download_all:
    input:
        expand(RAW_DIR + "{dataset}.mat", dataset=config["datasets"]), 
        expand(MODEL_OUTPUT_DIR + "{dataset}_inputInfo.mat", dataset=config["datasets"]), 
        expand(MODEL_OUTPUT_DIR + "{dataset}_{output_type}.h5", dataset=config["datasets"], output_type=OUTPUT_TYPES)

rule download_model:
    params:
         source = lambda wildcards: config["datasets"][wildcards.dataset][wildcards.output_type]
    output:
        MODEL_OUTPUT_DIR + "{dataset}_{output_type}.h5"
    shell:
        "scp -T {config[username]}@{params.source} {output}"

rule download_raw:
    params:
        source = lambda wildcards: config["datasets"][wildcards.dataset]["raw"]
    output:
        RAW_DIR + "{dataset}.mat"
    shell:
        "scp -T {config[username]}@{params.source} {output}"

rule download_inputInfo:
    params:
         source = lambda wildcards: config["datasets"][wildcards.dataset]["inputInfo"]
    output:
        MODEL_OUTPUT_DIR + "{dataset}_inputInfo.mat"
    shell:
        "scp -T {config[username]}@{params.source} {output}"

rule convert_pandas:
    input:
        RAW_DIR + "{dataset}.mat"
    output:
        INTERMEDIATE_DIR + "{dataset}.p"
    script:
        "src/convert_to_pandas.py"

#exploration notebooks
rule peak_analysis:
    input:
        "data/intermediate/rockstar.p",
        "data/model_output/rockstar_inputInfo.mat",
        "data/model_output/rockstar_valid.h5",
        SRC_DIR + "process_inputs.py",
    output:
        "notebooks/peak_analysis.ipynb"
    notebook:
        "notebooks/peak_analysis.ipynb"

rule integral_analysis:
    input:
        "data/intermediate/rockstar.p",
        "data/model_output/rockstar_inputInfo.mat",
        "data/model_output/rockstar_valid.h5",
        SRC_DIR + "process_inputs.py",
    output:
        "notebooks/integral_analysis.ipynb"
    notebook:
        "notebooks/integral_analysis.ipynb"

##Figures
rule notebook_to_html:
    input:
        "notebooks/{notebook}.ipynb",
        GIT_HEAD
    output:
        "notebooks/{notebook}.html"
    shell:
        "jupyter nbconvert --to html --no-input {input[0]}"