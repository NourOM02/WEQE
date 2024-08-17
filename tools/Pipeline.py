# Load parameters
import ujson
import os
config = f"{os.path.dirname(os.path.abspath(__file__))}/config.json"
with open(config, 'r') as json_file:
    config = ujson.load(json_file)
    datasets = config['datasets']
    expansions = config['expansions']
    project_path = config['project_path']
    cache_dir = config['cache_dir']
    raw_datasets = f"{cache_dir}/.raw_datasets"
    metrics = config['metrics']
    
from tools.Dataset import Dataset
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mpl

class Pipeline:
    def __init__(self) -> None:
        self.datasets:list[Dataset] = self._initialize_datasets()

    def _initialize_datasets(self):
        datasets_objects = []
        for dataset in datasets:
            datasets_objects.append(Dataset(dataset, "GROQ_KEY"))
        return datasets_objects

    def batch_to_json(self):
        """
        This function create json files that follow the BEIR format for each dataset
        in the specified config file.
        """
        for dataset in self.datasets:
            print(f"Creating json file for {dataset.name}...", end='', flush=True)
            if os.path.exists(dataset.json_path):
                print("(Json file already exists! Skipping...)")
                continue
            dataset.to_json()
            print("Done!")
        print("All json files created successfully!\n\n")

    def batch_index(self):
        """
        This function indexes all the datasets in the specified config file.
        """
        for dataset in self.datasets:
            print(f"Indexing {dataset.name}...", end='', flush=True)
            if not os.path.exists(dataset.index_path):
                dataset.index()
            print("Done!")
        print("All datasets indexed successfully!\n\n")

    def batch_retrieve(self):
        """
        This function retrieves relevant documents for queries of datasets specified
        by the config file
        """

        for dataset in self.datasets:
            print(f"Retrieving {dataset.name}...", end='', flush=True)
            for expansion in [None]+dataset.expanded:
                dataset.approach = expansion
                dataset.init_paths()
                print(f"{expansion}...", end='', flush=True)
                if not os.path.exists(dataset.run_path):
                    dataset.retrieve()
            print("Done!")
        print("All datasets are retrieved successfully!\n\n")

    def batch_examples(self):
        """
        This function samples examples for each dataset specified by the config file
        """
        for dataset in self.datasets:
            print(f"Sampling examples for {dataset.name} ...", end='', flush=True)
            if not os.path.exists(dataset.examples_path):
                dataset.examples()
            print("Done!")
        print("Examples are prepared for all datasets successfully!\n\n")

    def batch_PRF(self):
        """
        This function creates PRF file for each dataset specified by the config file
        """
        for dataset in self.datasets:
            print(f"Creating PRF for {dataset.name} ...", end='', flush=True)
            if not os.path.exists(dataset.PRF_path):
                dataset.PRF()
            print("Done!")

        print("PRF files are created for all datasets successfully!\n\n")

    def batch_expand(self):
        """
        This function expands queries for each dataset and for each method specified
        by the config file
        """
        for dataset in self.datasets:
            print(f"Expanding queries for {dataset.name} ...", end='', flush=True)
            dataset.expand(dataset.examples_path, dataset.PRF_path)
            dataset.assemble()
            print("Done!")
        print("Queries are expanded for all datasets successfully!\n\n")

    def batch_evaluate(self):
        """
        This function evaluates the performance of the specified datasets
        """
        results = {
            metric : {
                "dataset" : [dataset.name for dataset in self.datasets]
            } for metric in metrics
        }
        for dataset in self.datasets:
            print(f"Evaluating {dataset.name}...", end='', flush=True)
            
            for expansion in [None]+expansions:
                dataset.approach = expansion
                dataset.init_paths()
                output = dataset.evaluate()
                for metric in output:
                    if expansion not in results[metric]:
                        results[metric][expansion] = []
                    results[metric][expansion].append(output[metric])
            print("Done!")
        
        for metric in metrics:
            pd.DataFrame(results[metric]).to_csv(f"{cache_dir}/results/{metric}.csv", index=False)
        print("All datasets are evaluated successfully!\n\n")

    def execute(self):
        """
        """
        
        # Prepare datasets
        print("****************************************************************")
        print(f"Initial Preparation ...")
        print("****************************************************************")
        self.batch_to_json()
        self.batch_index()

        # Intial retrieve
        print("****************************************************************")
        print(f"Initial Retrieval...")
        print("****************************************************************")
        self.batch_retrieve()
        
        # Generate examples and PRF
        print("****************************************************************")
        print(f"Generating Examples and PRF...")
        print("****************************************************************")
        self.batch_examples()
        self.batch_PRF()
        
        # Expand queries
        print("****************************************************************")
        print(f"Expanding Queries...")
        print("****************************************************************")
        self.batch_expand()

        
        # Retrieve and evaluate for expanded queries
        print("****************************************************************")
        print(f"Retrieving and Evaluating for expanded queries...")
        print("****************************************************************")
        self.batch_retrieve()
        self.batch_evaluate()
