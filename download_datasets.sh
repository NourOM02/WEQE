#!/usr/bin/bash

directory=".raw_datasets"

if [ ! -d "$HOME/$directory" ]; then
	echo "Directory does not exist! Creating... "
	mkdir "$HOME/$directory"
	echo "Directory created. Starting downlaod..."

else
	echo "Directory already exists. Starting the downlaod..."
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
	[trec-covid]=https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/trec-covid.zip
	[webis-touche2020]=https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/webis-touche2020.zip
)

function exist_data () {
	dataset_name=$1
	if [ -f "$HOME/$directory/$dataset_name.zip" ]; then
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
	wget --directory-prefix $HOME/$directory $download_link
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
