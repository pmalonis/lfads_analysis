import fabric
from parse import parse
import yaml

servers = ["macleanlab@205.208.22.225", 
            "macleanlab@205.208.22.226", 
            "macleanlab@205.208.22.225",
            "macleanlab@205.208.22.226",
            "macleanlab@205.208.22.226"]
run_paths = ["~/peter/lfads_analysis/data/model_output/controller_params_sweep/",
            "~/peter/lfads_analysis/data/model_output/controller_params_sweep/",
            "~/peter/lfads_analysis/data/model_output/long_training",
            "~/peter/lfads_analysis/data/model_output/long_training",   
            "~/peter/lfads_analysis/data/model_output/controller_co_dim_sweep"]
param_prefixes = ["225","226","225","226","226"]

raw_data_path = "~/peter/lfads_analysis/data/raw/"
config_file = "config.yml"
output_file = "config.yml"

for server, run_path, param_prefix in zip(servers, run_paths, param_prefixes):

    connect = fabric.Connection(server)

    config = yaml.safe_load(open(config_file))

    if "datasets" not in config.keys():
        config["datasets"] = {}

    with connect.cd(run_path):
        param_dirs = connect.run("ls param_* -d").stdout.split()
        dset_dirs = connect.run("ls data_*/* -d").stdout.split()
        dset_names = [parse("single_{dset}", dset_dir.split('/')[1]).named["dset"] for dset_dir in dset_dirs]
        for dset_name,dset_dir in zip(dset_names, dset_dirs):
            if dset_name not in config["datasets"].keys():
                config["datasets"][dset_name] = {}
                config["datasets"][dset_name]["raw"] = server + ":" + raw_data_path + dset_name + ".mat"
                inputInfo_path = connect.run("ls data_*/single_%s/inputInfo_%s.mat"%(dset_name,dset_name)).stdout.split()[0]
                config["datasets"][dset_name]["inputInfo"] = server + ":" + run_path + "/" + inputInfo_path

            if "params" not in config["datasets"][dset_name].keys():
                config["datasets"][dset_name]["params"] = {}

            for param_dir in param_dirs:
                try:
                    train_path = connect.run("ls %s/single_%s/lfadsOutput/*h5_train*"%(param_dir,dset_name)).stdout.split()[0]
                    valid_path = connect.run("ls %s/single_%s/lfadsOutput/*h5_valid*"%(param_dir,dset_name)).stdout.split()[0]
                except:
                    continue

                param_hash = parse("param_{param_hash}", param_dir).named["param_hash"]
                param_hash = param_hash.replace("_", "~")
                param_hash = param_prefix + '_' + param_hash
                config["datasets"][dset_name]["params"][param_hash] = {}
                config["datasets"][dset_name]["params"][param_hash]["train"] = server + ":" + run_path + train_path
                config["datasets"][dset_name]["params"][param_hash]["valid"] = server + ":" + run_path + valid_path


    yaml.safe_dump(config, open(output_file,'w'))
