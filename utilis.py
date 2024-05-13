# Load parameters
import ujson
import os
config = f"{os.path.dirname(os.path.abspath(__file__))}/config.json"
with open(config, 'r') as json_file:
    config = ujson.load(json_file)
    raw_datasets = config['raw_datasets']

# Load libraries
import shutil
import zipfile
from tqdm import tqdm
import pandas as pd
from time import perf_counter as pf

def _unzip(dataset:str, path:str) -> tuple[list, bool]:
    paths = []
    with zipfile.ZipFile(path + ".zip", 'r') as zip_ref:
        if not os.path.exists(path):
            os.makedirs(path)
        zip_ref.extractall(os.path.dirname(path))
    sub_tree = True if len(os.listdir(path)) > 3 else False
    if sub_tree:
        for dir in os.listdir(path):
            shutil.move(f"{path}/{dir}", f"{raw_datasets}/{dataset}-{dir}")
            paths.append(f"{raw_datasets}/{dataset}-{dir}")
        os.rmdir(path)
    else:
        paths.append(path)
    return paths, sub_tree

def _to_json(paths):
    output_c = []
    output_q = {"id": [], "query": []}
    id = [1,1]
    for path in paths:
        mappings_c = {}
        mappings_q = {}
        with open(f"{path}/corpus.jsonl", 'r') as corpus:
            for line in tqdm(corpus):
                line = ujson.loads(line)
                obj = {"id": str(id[0]), "contents" : f"{line['text']} {line['title']}"}
                output_c.append(obj)
                mappings_c[str(line["_id"])] = str(id[0])
                id[0] += 1
        with open(f"{path}/queries.jsonl", 'r') as queries:
            for line in tqdm(queries):
                line = ujson.loads(line)
                mappings_q[str(line["_id"])] = str(id[1])
                output_q["id"].append(str(id[1]))
                output_q["query"].append(line["text"])
                id[1] += 1

        

        qrels = pd.read_csv(f"{path}/qrels/test.tsv", sep="\t")
        qrels[['query-id', 'corpus-id']] = qrels[['query-id', 'corpus-id']].astype(str)
        for i in range(len(qrels)):
            qrels.iloc[i, 0] = mappings_q[str(qrels.iloc[i, 0])]
            qrels.iloc[i, 1] = mappings_c[str(qrels.iloc[i, 1])]
    


    corpus = ujson.dumps(output_c, indent=4)
    queries = pd.DataFrame(output_q)

    if not os.path.exists("temp"):
        os.makedirs("temp")

    s = pf()
    with open("temp/output.json", "w") as json_file:
        json_file.write(corpus)
    print(f"Time taken: {pf() - s} seconds")

    queries.to_csv("queries.tsv", header=None, sep="\t", index=False)

    qrels.insert(1, 'tmp', pd.Series([0] * len(qrels)))
    qrels.to_csv("qrels.tsv", header=None, sep="\t", index=False)

    return

def to_json(dataset:str, zipped:bool=True) -> None:
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
    path = f"{raw_datasets}/{dataset}"
    sub_tree = False
    if zipped:
        paths,sub_tree = _unzip(dataset, path)

    _to_json(paths)

    return

def index(dataset, threads=4):
    """
    This function can be used after using 
    
    """
    command = f"python -m pyserini.index.lucene \
               --collection JsonCollection \
               --input temp \
               --index indexes/{dataset} \
               --generator DefaultLuceneDocumentGenerator \
               --threads {threads} \
               --storePositions --storeDocvectors --storeRaw"

    os.system(command)

def EDA(dataset):
    return

def retrieve(dataset):
    command = f"python -m pyserini.search.lucene \
            --threads 4 --batch-size 16 \
            --index indexes/{dataset} \
            --topics queries.tsv \
            --output runs/test.txt \
            --output-format trec \
            --b 0.75 --k1 1.2 \
            --hits 1000 --bm25 --remove-query"
    
    os.system(command)



def evaluate(dataset):
    command = f"python -m pyserini.eval.trec_eval -c -mrecip_rank.10 -mrecall.1000 -mmap -mndcg_cut.10 qrels.tsv\
            'runs/test.txt'"
    
    os.system(command)
