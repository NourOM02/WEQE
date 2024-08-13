# Load parameters
import ujson
import os
config = f"{os.path.dirname(os.path.abspath(__file__))}/config.json"
with open(config, 'r') as json_file:
    config = ujson.load(json_file)
    datasets = config['datasets']
    project_path = config['project_path']
    cache_dir = config['cache_dir']
    raw_datasets = f"{cache_dir}/.raw_datasets"
    expansions = config['expansions']
    metrics = config['metrics']


# Load libraries
import zipfile
from tqdm import tqdm
import subprocess
import pandas as pd
from tools.Expansion import Expansion
import re
import shutil

class Dataset(Expansion):
    def __init__(self, name:str, key) -> None:
        self.name = name
        super().__init__(self.name, key)
        self.raw_path = f"{raw_datasets}/{self.name}.zip"
        self.index_path = f"{cache_dir}/indexes/{self.name}/"
        self.json_path = f"{cache_dir}/.jsons/{self.name}/tmp.json"
        self.qrels_path = f"{cache_dir}/datasets/qrels/{self.name}.tsv"
        self.corpus_path = f"{cache_dir}/.jsons/{self.name}/tmp.json"
        self.examples_path = f"{cache_dir}/datasets/examples/{self.name}.tsv"
        self.PRF_path = f"{cache_dir}/datasets/PRF/{self.name}.json"

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
        print("1")
        queries = pd.read_csv(self.queries_path, sep="\t", header=None)
        qrels = pd.read_csv(self.qrels_path, sep="\t", header=None)
        corpus = ujson.load(open(self.corpus_path, 'r'))
        print("2")
        # filter qrels to keep only relevant passages
        qrels = qrels[qrels.iloc[:,3] != 0]
        
        # check if K is greater than the available sources
        K = min(K, len(corpus), len(queries), len(qrels))
        # sample K examples
        qrels = qrels.sample(K)
        # initialize a dictionary to store the examples
        dict = {"query" : [], "passage" : []}
        for i in range(len(qrels)):
            # determine the query_id and passage_id to use
            qid = qrels.iloc[i, 0]
            pid = qrels.iloc[i, 2]
            # append the query and passage to the dictionary
            dict["query"].append(queries[queries.iloc[:,0] == qid].iloc[:,1].values[0])
            dict["passage"].append(corpus[pid-1]["contents"])
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

        print(os.path.dirname(self.json_path))
        print(self.index_path)

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
    
        subprocess.run(command, shell=True)

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
        shutil.rmtree(f"{raw_datasets}/{self.name}")

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