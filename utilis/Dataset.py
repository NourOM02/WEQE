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


# Load libraries
import shutil
import zipfile
from tqdm import tqdm
import subprocess
import pandas as pd
from Expansion import Expansion
import re
import string
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mpl
from time import perf_counter as pf
from wordcloud import WordCloud
from stopwords import stop_words as stopwords

class Dataset(Expansion):
    def __init__(self, name:str, key) -> None:
        self.name = name
        super().__init__(self.name, key)
        self.expansions = expansions.copy()
        self.raw_path = f"{raw_datasets}/{self.name}.zip"
        self.index_path = f"{dependencies}/indexes/{self.name}/"
        self.json_path = f"{dependencies}/.jsons/{self.name}/tmp.json"
        self.qrels_path = f"{dependencies}/datasets/qrels/{self.name}.tsv"
        self.corpus_path = f"{dependencies}/.jsons/{self.name}/tmp.json"
        self.examples_path = f"{dependencies}/datasets/examples/{self.name}.tsv"
        self.PRF_path = f"{dependencies}/datasets/PRF/{self.name}.json"
        self._dependencies()
        
    def _dependencies(self):
        """
        This function is used to create the necessary directories for the datasets
        """

        needed_folders = [dependencies, dependencies+"/indexes", dependencies+"/.jsons",
                          dependencies + "/.runs", dependencies + "/results",
                          dependencies+"/datasets", dependencies+"/datasets/queries",
                          dependencies+"/datasets/qrels", dependencies+"/datasets/examples",
                          dependencies+"/datasets/PRF"]
        
        for folder in needed_folders:
            if not os.path.exists(folder):
                os.makedirs(folder)

    def _unzip(self) -> None:
        # Check if the dataset is already unzipped
        if not os.path.exists(self.raw_path[:-4]):
            print(f"Unzipping {self.name}...", end="", flush=True)
            # Unzip the dataset in the raw_datasets directory
            with zipfile.ZipFile(self.raw_path, 'r') as zip_ref:
                if not os.path.exists(self.raw_path[:-4]):
                    os.makedirs(self.raw_path[:-4])
                zip_ref.extractall(os.path.dirname(self.raw_path[:-4]))
            if self.name == "msmarco":
                os.remove(f"{self.raw_path[:-4]}/qrels/test.tsv")
                os.rename(f"{self.raw_path[:-4]}/qrels/dev.tsv",
                          f"{self.raw_path[:-4]}/qrels/test.tsv")

    def _to_json(self) -> None:
        output_c = []
        output_q = {"id": [], "query": []}
        qrels = pd.DataFrame({'query-id': [], 'corpus-id': [], 'score': []})
        # Initalize corpus and queries ids
        id = [1,1]
        # determine the paths depending on whether there are subdirectories
        path_content = [path for path in os.listdir(self.raw_path[:-4])]
        paths = [f"{self.raw_path[:-4]}/{sub}" for sub in path_content]\
                if len(path_content) > 3 else [self.raw_path[:-4]]
    
        for path in paths:
            mappings_c = {}
            mappings_q = {}
            # Preapre the corpus for the given path
            with open(f"{path}/corpus.jsonl", 'r') as corpus:
                for line in tqdm(corpus):
                    line = ujson.loads(line)
                    obj = {"id": str(id[0]), "contents" : f"{line['text']} {line['title']}"}
                    output_c.append(obj)
                    mappings_c[str(line["_id"])] = str(id[0])
                    id[0] += 1
            # Prepare the queries for the given path
            with open(f"{path}/queries.jsonl", 'r') as queries:
                for line in tqdm(queries):
                    line = ujson.loads(line)
                    mappings_q[str(line["_id"])] = str(id[1])
                    output_q["id"].append(str(id[1]))
                    output_q["query"].append(line["text"])
                    id[1] += 1
            # Prepare the qrels for the given path
            qrels_i = pd.read_csv(f"{path}/qrels/test.tsv", sep="\t")
            qrels_i[['query-id', 'corpus-id']] = qrels_i[['query-id', 'corpus-id']].astype(str)
            """
            This part of the code is modified to deal with a problem found in the
            ArguAna dataset. The problem is that the qrels file contains a corpus
            id that is not found in the corpus file. To avoid such a problem, a 
            missings variable is created and is filled with the indices of the rows
            that failed : (query-id in query file) AND (corpus-is in corpus file)
            """
            missings = []
            for i in range(len(qrels_i)):
                # query-id and corpus-id verification
                if qrels_i.iloc[i, 0] not in mappings_q or qrels_i.iloc[i, 1] not in mappings_c:
                    missings.append(i)
                    continue
                qrels_i.iloc[i, 0] = mappings_q[str(qrels_i.iloc[i, 0])]
                qrels_i.iloc[i, 1] = mappings_c[str(qrels_i.iloc[i, 1])]
            # delete missings from the file
            qrels_i.drop(missings, inplace=True)
            qrels = pd.concat([qrels, qrels_i], ignore_index=True)

        # check if folder exists
        if not os.path.exists(os.path.dirname(self.json_path)):
            os.makedirs(os.path.dirname(self.json_path))

        # Save corpus     
        corpus = ujson.dumps(output_c, indent=4)
        with open(self.json_path, "w") as json_file:
            json_file.write(corpus)

        # Save queries
        queries = pd.DataFrame(output_q)
        queries = queries[queries["id"].isin(qrels.iloc[:,0])]
        queries.to_csv(self.queries_path, header=None, sep="\t", index=False)

        # Save qrels
        n = len(qrels)
        fill_column = [0] * n
        qrels.insert(1, 'tmp', fill_column)
        qrels.to_csv(self.qrels_path, header=None, sep="\t", index=False)

        # Delete the unzipped dataset
        #shutil.rmtree(self.raw_path[:-4])

    def to_json(self):
        """
        The use case for this function is to prepare the dataset for indexing using pyserini.
        The function assumes the BEIR format :\n
            \tthe dataset is zipped and it contains the following files\n
                \t\t- corpus.jsonl
                \t\t- queries.jsonl
                \t\t- qrels/dev.tsv, train.tsv, test.tsv
        The output is a json file that contains the corpus in the following format:\n
            \t{\n
                \t\t"id": "id",\n
                \t\t"contents": "text"\n
            \t}\n
        It is saved in a temporary directory 'temp' as 'output.json'

        Parameters:
        ----------

        path: str
            The path to the zipped dataset.
        """

        self._unzip()
        self._to_json()

    def examples(self, K:int=1000):
        """
        This function prepare query-passage pairs examples for the dataset.

        Parameters:
        ----------
        K: int
            (default: 1000) The number of examples to prepare. 
        """
        queries = pd.read_csv(self.queries_path, sep="\t", header=None)
        qrels = pd.read_csv(self.qrels_path, sep="\t", header=None)
        corpus = ujson.load(open(self.corpus_path, 'r'))
        
        # filter qrels to keep only relevant passages
        qrels = qrels[qrels.iloc[:,3] == 1]
        
        # check if K is greater than the available sources
        K = min(K, len(corpus), len(queries), len(qrels))
        # sample K examples
        qrels = qrels.sample(K)
        # initialize a dictionary to store the examples
        dict = {"query" : [], "passage" : []}
        for i in range(len(qrels)):
            # determine the query_id and passage_id to use
            qid = qrels.iloc[i, 0]
            pid = qrels.iloc[i, 0]
            # append the query and passage to the dictionary
            dict["query"].append(queries[queries.iloc[:,0] == qid].iloc[:,1].values[0])
            dict["passage"].append(corpus[pid-1])
        # save the examples to a tsv file
        df = pd.DataFrame(dict)
        df.to_csv(self.examples_path, sep='\t', index=False, header=False)

    def index(self, threads=8):
        """
        This function can be used to index the dataset that is already in the format
        specified by pyserini. It takes as input the path to the directory containing
        the file to be indexed. If you are using the BEIR dataset, you can use the
        to_json() function to prepare the dataset for indexing. If not, you should
        format your dataset first.Refer to the following link for more details :
        https://github.com/castorini/pyserini/blob/master/docs/usage-index.md#building\
-a-bm25-index-direct-java-implementation

        Parameters:
        ----------

        input: str
            (default: "temp") The path to the directory containing the file to be indexed.
        
        threads: int
            (default: 8) The number of threads to be used for indexing.
            
        """

        command = f"python -m pyserini.index.lucene \
               --collection JsonCollection \
               --input {os.path.dirname(self.json_path)} \
               --index {self.index_path} \
               --generator DefaultLuceneDocumentGenerator \
               --threads {threads} \
               --storePositions --storeDocvectors --storeRaw"

        run = subprocess.run(command, shell=True)

    def retrieve(self):
        """
        This function can be used to retrieve relevant documents for the queries of the
        dataset. The function also assumes that the dataset is already indexed. If not,
        you should index the dataset first.
        """
        command = f"python -m pyserini.search.lucene \
                --threads 4 --batch-size 16 \
                --index {self.index_path} \
                --topics {self.queries_path} \
                --output {self.run_path} \
                --output-format trec \
                --b 0.75 --k1 1.2 \
                --hits 1000 --bm25 --remove-query"
    
        os.system(command)

    def PRF(self, k:int=5):
        print(f"generating PRF for {self.name}")
        queries = pd.read_csv(self.queries_path, sep="\t", header=None)
        corpus = ujson.load(open(self.corpus_path, 'r'))
        runs = pd.read_csv(self.run_path, sep=" ", header=None)

        PRF = {
            key : [
                corpus[doc-1]["contents"]
                for doc in runs[runs.iloc[:,0] == key].iloc[:,2][:k]
            ]
            for key in queries.iloc[:,0]}
        
        ujson.dump(PRF, open(self.PRF_path, 'w'))

    def evaluate(self):
        results = {key : 0 for key in metrics}
    
        # evaluate for MRR@10 , R@1000,  MAP and NDCG@10
        command = f"python -m pyserini.eval.trec_eval -c -mrecip_rank.10 -mrecall.1000\
                    -mmap -mndcg_cut.10 {self.qrels_path} {self.run_path}"
        
        run = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # extract metrics from stdout
        patterns = {key : fr"{metrics[key]}\s+all\s+(\d+\.\d+)" for key in metrics}
        for key, pattern in patterns.items():
            match = re.search(pattern, run.stdout.decode("utf-8"))
            if match:
                results[key] = float(match.group(1))*100

        return results
    
    def _numbers(self):
        """
        This function is used to determine the number/lengths of queries, passages and qrels
        in the dataset. It is used to determine the number of examples to prepare.
        """
        output = {"cardinal" : [], "length" : [], "ratio" : 0}
        queries = pd.read_csv(self.queries_path, sep="\t", header=None)
        queries['word_count'] = queries.iloc[:,1].apply(lambda x: len(x.split()))
        qrels = pd.read_csv(self.qrels_path, sep="\t", header=None)
        corpus = pd.read_json(self.corpus_path)
        words = {}
        def clean_text(text):
            # Remove punctuation
            text = text.translate(str.maketrans('', '', string.punctuation))
            # Remove numbers
            text = re.sub(r'\d+', '', text)
            # Remove extra whitespace
            text = text.strip()
            return text
        stopwords_ = set(stopwords)
        for i in tqdm(range(len(corpus))):
            for word in clean_text(corpus.iloc[i,1].lower()).split():
                if word in stopwords_:
                    continue
                if word in words:
                    words[word] += 1
                else:
                    words[word] = 1

        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(words)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')  # No axes for the word cloud
        plt.savefig(f"{self.name}.png")
        plt.show()

        corpus['word_count'] = corpus.iloc[:,1].apply(lambda x: len(x.split()))
        queries["word_count"] = queries.iloc[:,1].apply(lambda x: len(x.split()))


        histogram = plt.figure(figsize=(18, 6), dpi=300)
        gs = GridSpec(1, 2, figure=histogram)

        ax1 = histogram.add_subplot(gs[0, 0])
        ax2 = histogram.add_subplot(gs[0, 1])

        # Colors and normalization for bar colors
        colours = ["#bbdefb", "#2196f3"]
        for i, ax in enumerate([ax1, ax2]):
            active = corpus if ax == ax1 else queries

            # Grid customization
            ax.grid(which="major", axis='x', color='#DAD8D7', alpha=0.5, zorder=1)
            ax.grid(which="major", axis='y', color='#DAD8D7', alpha=0.5, zorder=1)

            # Plotting the bars
            hist = ax.hist(active["word_count"], bins=20, color="#2196f3")
            ax.set_title(f'Distribution of word count in {"Corpus" if ax == ax1 else "Queries"}')
            ax.set_xlabel('word count')
            ax.set_ylabel('number of occurences')
            ax.tick_params(axis='x', rotation=45)

        # Show the plot
        histogram.tight_layout()
        #plt.savefig(f'{self.name}')
        plt.show()

        output["cardinal"] = [len(queries), len(corpus), len(qrels)]
        output["length"] = [round(queries['word_count'].mean(), 2),
                            round(corpus['word_count'].mean(), 2)]
        output["ratio"] = round(len(qrels)/len(queries), 2)
        
        return output