# Load parameters
import ujson
import os
config = f"{os.path.dirname(os.path.abspath(__file__))}/config.json"
with open(config, 'r') as json_file:
    config = ujson.load(json_file)
    datasets = config['datasets']

from utilis import Dataset
import os

# Prepare and index datasets
for dataset_name in datasets:
    dataset = Dataset(dataset_name)
    #dataset.to_json()
    #dataset.index()
    dataset.retrieve()