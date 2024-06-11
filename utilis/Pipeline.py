# Load parameters
import ujson
import os
config = f"{os.path.dirname(os.path.abspath(__file__))}/config.json"
with open(config, 'r') as json_file:
    config = ujson.load(json_file)
    raw_datasets = config['raw_datasets']
    datasets = config['datasets']
    expansions = config['expansions']
    project_path = config['project_path']
    dependencies = config['dependencies']
    metrics = config['metrics']
    
from Dataset import Dataset
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
            datasets_objects.append(Dataset(dataset, "GROQ_KEY_1"))
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
            for expansion in [None]+dataset.expanded:
                print(f"Retrieving {dataset.name}...", end='', flush=True)
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
            for expansion in [None]+dataset.expansions:
                dataset.approach = expansion
                print(dataset.approach)
                dataset.init_paths()
                print(dataset.queries_path)
                output = dataset.evaluate()
                for metric in output:
                    print(output[metric])
                    if expansion not in results[metric]:
                        results[metric][expansion] = []
                    results[metric][expansion].append(output[metric])
            print("Done!")
        
        for metric in metrics:
            pd.DataFrame(results[metric]).to_csv(f"{dependencies}/results/{metric}.csv", index=False)
        print("All datasets are evaluated successfully!\n\n")                    

    def EDA(self):
        """
        This function performs exploratory data analysis on the datasets
        """

        def format_number(num):
            if num >= 1_000_000:
                return f'{num / 1_000_000:.1f}M'
            elif num >= 1_000:
                return f'{num / 1_000:.1f}K'
            else:
                return str(num)
        
        numbers = {'queries':[], 'corpus':[], 'qrels':[]}
        lengths = {'queries':[], 'corpus':[]}
        ratios = []

        if not os.path.exists(f"{dependencies}/analysis"):
            os.makedirs(f"{dependencies}/analysis")

        existance = os.path.exists(f"{dependencies}/analysis/stats.json")

        saved = ujson.load(open(f"{dependencies}/analysis/stats.json", 'r'))\
            if existance else {} 

        for dataset in self.datasets:
            print(f"Processing {dataset.name}...")
            stats = dataset._numbers() if not existance else saved[dataset.name]
            saved[dataset.name] = stats
            for i, key in enumerate(['queries', 'corpus', 'qrels']):
                numbers[key].append(stats["cardinal"][i])
            for i, key in enumerate(['queries', 'corpus']):
                lengths[key].append(stats["length"][i])
            ratios.append(stats["ratio"])
            if not existance:
                with open(f"{dependencies}/analysis/stats.json", 'w') as json_file:
                    ujson.dump(saved, json_file)

        ##### Figure 1 #####

        fig1 = plt.figure(figsize=(18, 12), dpi=300)
        gs = GridSpec(2, 4, figure=fig1, height_ratios=[1, 1], width_ratios=[1, 1, 1, 1])

        ax1 = fig1.add_subplot(gs[0, 0:2])
        ax2 = fig1.add_subplot(gs[0, 2:4])
        ax3 = fig1.add_subplot(gs[1, 1:3])

        for i, ax in enumerate([ax1, ax2, ax3]):
            active = list(numbers.keys())[i]
            colours = ["#bbdefb", "#2196f3"]
            cmap = mpl.LinearSegmentedColormap.from_list("colour_map", colours, N=256)
            norm = mpl.Normalize(min(numbers[active]),\
                                 max(numbers[active]))
            ax.grid(which="major", axis='x', color='#DAD8D7', alpha=0.5, zorder=1)
            ax.grid(which="major", axis='y', color='#DAD8D7', alpha=0.5, zorder=1)
            bar = ax.bar([dataset.name for dataset in self.datasets], numbers[active],\
                         color=cmap(norm(numbers[active])))
            ax.set_title(f'Number of {active}')
            ax.set_xlabel('Dataset')
            ax.set_ylabel(f'#{active}')
            ax.tick_params(axis='x', rotation=45)
            ax.bar_label(bar, labels=[format_number(e) for e in numbers[active]]\
                          ,padding=3, color='black', fontsize=8)

        fig1.tight_layout()
        plt.savefig('fig1_numbers_distribution.png')

        ##### Figure 2 #####

        # Plot 2: Distribution of lengths of queries and corpus documents (unchanged)
        dataset_names = [dataset.name for dataset in self.datasets]

        fig2 = plt.figure(figsize=(18, 6), dpi=300)
        gs = GridSpec(1, 2, figure=fig2, width_ratios=[1, 1])

        ax1 = fig2.add_subplot(gs[0, 0])
        ax2 = fig2.add_subplot(gs[0, 1])

        for i, ax in enumerate([ax1, ax2]):
            active = list(lengths.keys())[i]
            colours = ["#bbdefb", "#2196f3"]
            cmap = mpl.LinearSegmentedColormap.from_list("colour_map", colours, N=256)
            norm = mpl.Normalize(vmin=min(lengths[active]), vmax=max(lengths[active]))
            
            ax.grid(which="major", axis='x', color='#DAD8D7', alpha=0.5, zorder=1)
            ax.grid(which="major", axis='y', color='#DAD8D7', alpha=0.5, zorder=1)
            
            bar = ax.bar(dataset_names, lengths[active], color=cmap(norm(lengths[active])))
            ax.set_title(f'Average length of {active} {"documents" if active == "corpus" else ""} (words)')
            ax.set_xlabel('Dataset')
            ax.set_ylabel(f'Length (words)')
            ax.tick_params(axis='x', rotation=45)
            
            ax.bar_label(bar, labels=[format_number(e) for e in lengths[active]], padding=3, color='black', fontsize=8)

        fig2.tight_layout()
        plt.savefig('fig2_lengths_distribution.png')
        plt.show()

        ##### Figure 3 #####

        fig3 = plt.figure(figsize=(9, 6), dpi=300)
        gs = GridSpec(1, 1, figure=fig3)

        ax = fig3.add_subplot(gs[0, 0])

        # Colors and normalization for bar colors
        colours = ["#bbdefb", "#2196f3"]
        cmap = mpl.LinearSegmentedColormap.from_list("colour_map", colours, N=256)
        norm = mpl.Normalize(vmin=min(ratios), vmax=max(ratios))

        # Grid customization
        ax.grid(which="major", axis='x', color='#DAD8D7', alpha=0.5, zorder=1)
        ax.grid(which="major", axis='y', color='#DAD8D7', alpha=0.5, zorder=1)

        # Plotting the bars
        bar = ax.bar(dataset_names, ratios, color=cmap(norm(ratios)))
        ax.set_title('Qrels/Queries Ratio for Each Dataset')
        ax.set_xlabel('Dataset Names')
        ax.set_ylabel('Ratio')
        ax.tick_params(axis='x', rotation=45)

        # Formatting bar labels with the format_number function
        ax.bar_label(bar, labels=[format_number(e) for e in ratios], padding=3, color='black', fontsize=8)

        fig3.tight_layout()
        plt.savefig('fig3_ratios.png')
        plt.show()

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
        #self.batch_expand()
        
        # Retrieve and evaluate for expanded queries
        print("****************************************************************")
        print(f"Retrieving and Evaluating for expanded queries...")
        print("****************************************************************")
        #self.batch_retrieve()
        #self.batch_evaluate()

x = Pipeline()
x.execute()