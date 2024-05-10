import utilis
import os

dataset = "msmarco"
zip_path = "/home/application/.raw_datasets/msmarco.zip"
json_path = f"{dataset}/output.json"
#utilis.prepare_dataset(zip_path, dataset)
#utilis.index(dataset)
#utilis.retrieve(dataset)
#utilis.evaluate(dataset)
utilis.expand(json_path)