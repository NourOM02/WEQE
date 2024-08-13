#!/usr/bin/bash

# Determine the directory where the script is located
script_dir=$(realpath "$0")
folder=$(dirname "$script_dir")
cache_dir="$(dirname "$folder")/.WEQE"

# Create cache directory it not exist
if [ ! -d $cache_dir ]; then
	mkdir $cache_dir
fi


# Create python environment if not exist
if [ ! -d "$cache_dir/venv" ]; then
    python -m venv "$cache_dir/venv"
fi

# Activate python environment
source "$cache_dir/venv/bin/activate"

# Install required packages
pip install -r "$folder/requirements.txt"

# Install java
sudo apt update
sudo apt install openjdk-11-jdk
sudo apt install openjdk-11-jre

# Create .jsons sub directory if not exist
if [ ! -d "$cache_dir/.jsons" ]; then
    mkdir "$cache_dir/.jsons"
fi

# Create .runs sub directory if not exist
if [ ! -d "$cache_dir/.runs" ]; then
    mkdir "$cache_dir/.runs"
fi


# Create datasets sub directory if not exist
if [ ! -d "$cache_dir/datasets" ]; then
    mkdir "$cache_dir/datasets"
fi

# Create examples sub sub directory if not exist
if [ ! -d "$cache_dir/datasets/examples" ]; then
    mkdir "$cache_dir/datasets/examples"
fi

# Create queries sub sub directory if not exist
if [ ! -d "$cache_dir/datasets/queries" ]; then
    mkdir "$cache_dir/datasets/queries"
fi

# Create qrels sub sub directory if not exist
if [ ! -d "$cache_dir/datasets/qrels" ]; then
    mkdir "$cache_dir/datasets/qrels"
fi

# Create PRF sub sub directory if not exist
if [ ! -d "$cache_dir/datasets/PRF" ]; then
    mkdir "$cache_dir/datasets/PRF"
fi

# Create indexes sub directory if not exist
if [ ! -d "$cache_dir/indexes" ]; then
    mkdir "$cache_dir/indexes"
fi

# Create logs sub directory if not exist
if [ ! -d "$cache_dir/logs" ]; then
    mkdir "$cache_dir/logs"
fi

# Create results sub directory if not exist
if [ ! -d "$cache_dir/results" ]; then
    mkdir "$cache_dir/results"
fi

raw_datasets="$cache_dir/.raw_datasets"

if [ ! -d $raw_datasets ]; then
	mkdir $raw_datasets
fi

declare -A datasets=(
	[arguana]=https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/arguana.zip
	[climate-fever]=https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/climate-fever.zip
	[cqadupstack]=https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/cqadupstack.zip
	[dbpedia-entity]=https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/dbpedia-entity.zip
	[fever]=https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fever.zip
	[fiqa]=https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip
	[hotoptqa]=https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/hotpotqa.zip
	[msmarco]=https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/msmarco.zip
	[nfcorpus]=https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip
	[nq]=https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nq.zip
	[quora]=https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/quora.zip
	[scidocs]=https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scidocs.zip
	[scifact]=https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip
	[trec-covid]=https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/trec-covid.zip
	[webis-touche2020]=https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/webis-touche2020.zip
)

function exist_data () {
	dataset_name=$1
	if [ -f "$raw_datsets/$dataset_name.zip" ]; then
		return 0
	else
		return 1
	fi
}


downlaod_dataset() {
	dataset_name=$1
	download_link=$2
	echo $downlaod_link
	echo "Downloading $dataset_name..."
	wget --directory-prefix $raw_datasets $download_link
	echo "Download is complete."
}

for dataset in "${!datasets[@]}"; do
    dataset_name=$dataset
    download_link=${datasets[$dataset]}

	if exist_data "$dataset_name"; then
		echo "$dataset_name exists. Skipping."
	else
		downlaod_dataset "$dataset_name" "$download_link"
	fi
	
done
