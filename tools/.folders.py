import ujson
import os
config = f"{os.path.dirname(os.path.abspath(__file__))}/config.json"
with open(config, 'r') as json_file:
    config = ujson.load(json_file)
    cache_dir = config['cache_dir']

def create_folders():
    """
    This function is used to create the folders required for the project.
    """

    base_folders = ["indexes", ".jsons", ".runs", "results", "datasets", ".raw_datasets", "logs"]
    dataset_subfolders = ["queries", "qrels", "examples", "PRF"]
    
    folders = [os.path.join(cache_dir, folder) for folder in base_folders]
    folders += [os.path.join(cache_dir, "datasets", subfolder) for subfolder in dataset_subfolders]
    
    for folder in folders:
        os.makedirs(folder, exist_ok=True)

create_folders()