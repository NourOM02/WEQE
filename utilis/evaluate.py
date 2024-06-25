# Load parameters
import ujson
import os
config = f"{os.path.dirname(os.path.abspath(__file__))}/config.json"
with open(config, 'r') as json_file:
    config = ujson.load(json_file)
    raw_datasets = config['raw_datasets']
    datasets = config['datasets']
    project_path = config['project_path']
    dependencies = config['dependencies']
    metrics = config['metrics']
    expansions = config['expansions']

from Dataset import Dataset
import pandas as pd
from tqdm import tqdm

for approach in ["Q2D_ZS"]:
    for dataset in datasets[::-1]:
        if dataset != "scidocs":
            continue
        print(f"Evaluation of {dataset}")
        x = Dataset(dataset, "GROQ_KEY_1")
        result = x.evaluate()
        results = {"Expansion" : ["Baseline"]}
        for i, metric in enumerate(metrics):
            results[metric] = [result[metric]]
        x.approach = approach
        results["Expansion"].append(x.approach)
        x.init_paths()
        if not os.path.exists(f"{dependencies}/.runs/{x.name}_{approach}.txt"):
            x.retrieve()
        result = x.evaluate()
        for i, metric in enumerate(metrics):
            results[metric].append(result[metric])
        for key in ["1_4", "1_6", "1_8", "2_4", "2_6", "2_8", "3_4", "3_6", "3_8", "5_4", "5_6", "5_8"]:
            x.approach = approach + '_' + key
            results["Expansion"].append(x.approach)
            x.init_paths()
            #if not os.path.exists(f"{dependencies}/.runs/{x.name}_{x.approach}.txt"):
            x.retrieve()
            result = x.evaluate()
            for i, metric in enumerate(metrics):
                results[metric].append(result[metric])
        
        results = pd.DataFrame(results)
        results.to_csv(f"{dependencies}/results/{x.name}_{approach}.csv", index=False)
    