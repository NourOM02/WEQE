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

# approaches = ["Q2D", "Q2D_PRF", "Q2D_ZS"]
approaches = ["Q2D_ZS"]

from Dataset import Dataset

for approach in approaches:
    for dataset in datasets[::-1]:
        if dataset in ["trec-covid", "webis-touche2020", "hotpotqa", "nfcorpus", "nq", "quora", "scidocs", "scifact", ]:
            print(f"Offline expansion of {dataset}")
            msmarco = Dataset(dataset, "GROQ_KEY_1")
            while True:
                try:
                    msmarco.offline_expand(approach, msmarco.PRF_path)
                    break
                except Exception as e:
                    print(e)
                    continue