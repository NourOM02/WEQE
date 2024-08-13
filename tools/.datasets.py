import ujson
import requests
import os
from tqdm import tqdm
config = f"{os.path.dirname(os.path.abspath(__file__))}/config.json"
with open(config, 'r') as json_file:
    config = ujson.load(json_file)
    cache_dir = config['cache_dir']
    datasets = config['datasets']

raw_datasets = os.path.join(cache_dir, '.raw_datasets')

datasets_dict = {
    'arguana': 'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/arguana.zip',
    'climate-fever': 'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/climate-fever.zip',
    'cqadupstack': 'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/cqadupstack.zip',
    'dbpedia-entity': 'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/dbpedia-entity.zip',
    'fever': 'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fever.zip',
    'fiqa': 'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip',
    'hotpotqa': 'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/hotpotqa.zip',
    'msmarco': 'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/msmarco.zip',
    'nfcorpus': 'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip',
    'nq': 'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nq.zip',
    'quora': 'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/quora.zip',
    'scidocs': 'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scidocs.zip',
    'scifact': 'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip',
    'trec-covid': 'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/trec-covid.zip',
    'webis-touche2020': 'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/webis-touche2020.zip'
}

def exist_data(dataset_name):
    return os.path.isfile(os.path.join(raw_datasets, f"{dataset_name}.zip"))

def download_dataset(dataset_name, download_link):
    print(f"Downloading {dataset_name}...")
    response = requests.get(download_link, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kilobyte
    t = tqdm(total=total_size, unit='iB', unit_scale=True)
    with open(os.path.join(raw_datasets, f"{dataset_name}.zip"), 'wb') as f:
        for data in response.iter_content(block_size):
            t.update(len(data))
            f.write(data)
    t.close()
    if total_size != 0 and t.n != total_size:
        print("ERROR: Something went wrong")
    print("Download is complete.")


for dataset_name, download_link in datasets_dict.items():
    if dataset_name not in datasets:
        continue
    if exist_data(dataset_name):
        print(f"{dataset_name} exists. Skipping.")
    else:
        download_dataset(dataset_name, download_link)
