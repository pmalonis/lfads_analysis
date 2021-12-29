# lfads_analysis

This is the repository for code used in the analysis used in the paper,

>M1 dynamics share similar inputs for initiating and correcting movement
>Peter J. Malonis, Nicholas G. Hatsopoulos, Jason N. MacLean, Matthew T. Kaufman
>bioRxiv 2021.10.18.464704; doi: [https://doi.org/10.1101/2021.10.18.464704](https://doi.org/10.>1101/2021.10.18.464704).

## Installation
1. Install [miniconda](https://docs.conda.io/en/latest/miniconda.html)
2. Clone the repository:
    ```
    git clone https://github.com/pmalonis/lfads_analysis
    ```
3. Create the conda environment:
    ```
    cd lfads_analysis
    conda env create -f environment.yml
    ```

<!---
TODO: add info about setting up datasets
-->
## Setting up the configuration

`config.yml` contains the file paths to the data and lfads runs. Make the following edits to the file to specifiy a different LFADS run: 

1. `lfads_file_server` should specify the hostname of the server that stores the LFADS runs (you'll need to set up [public key authorization](https://www.cyberciti.biz/faq/how-to-set-up-ssh-keys-on-linux-unix/) to that server). 
2. `username` should be the username used to login to the server
3. `lfads_dir_path` should be the path to the directory that holds the LFADS runs on the server
4. `lfads_run_names` should be a list of the names of LFADS runs to use. Each should be a subdirectory of the `lfads_dir_path` directory

## Running the code
This analysis uses the [Snakemake](https://snakemake.readthedocs.io/en/stable/) workflow management system. To create the analysis figures, run the following:

```
snakemake -j1 make_all_figures
```