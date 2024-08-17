# Weighted Embeddings Query Expansion (WEQE)

WEQE is about Query Expansion using Generative AI. It applies 3 expansion methods from the following [paper](https://arxiv.org/pdf/2305.03653), in addition to introducing a novel expansion method: WEQE that outperforms other methods in Recall. It also provides tools for evaluating and comparing the results. A detailed report about the project can be found in the following [report](https://drive.google.com/file/d/1Bl_JYKQ7iOfnkXL9Q1fitUGf997mo5JT/view?usp=drive_link).

## Dependencies

The pipeline is based on pyserini, which requires the some dependencies to work properly. In addition to that, the BEIR benchmarking datasets are required. If you are on linux, the simplest way to setup all dependencies is by running the `environment.sh` script that takes care of all dependecies _(if you use the script, jump directly to <a href="#config">Config file</a>)_. However, if you are on windows, or you want to setup the dependencies manually, you may follow this guide (_It is a best practice to create a seperate folder `.WEQE` that will store the dependencies_).

**Java 11**

Pyserini works best with Java 11, if you are on windows you can download it following this [link](https://www.oracle.com/java/technologies/downloads/#license-lightbox). On Linux, you can simply run these commands:
    
    sudo apt update
    sudo apt install openjdk-11-jdk
    sudo apt install openjdk-11-jre

**Python packages**

To ensure a clean environment and to avoid versions problems, it is best to start by creating a virtual environment. Use VS code venv builder, or run the following command:

    python -m venv ".WEQE/venv"

Make sure to avtivate the environment before moving on the next step. On linux run :

    source ".WEQE/venv/bin/activate"

On windows run :

    .WEQE\venv\Scripts\activate

Once the environment is ready, install the packages provided in the `requirements.txt` file:

    pip install -r "$folder/requirements.txt"

**Needed folders**

Files are organized in folders to ensure a clean and user-friendly experience. Creating this folders before running the running the code is mandatory. Please run the `folders.py` script in the tools directory.

**Downloading BEIR datasets**

You can downlad the BEIR datasets manually following this [link](https://github.com/beir-cellar/beir), or you can the `datasets.py` script in the tools directory.

<strong id="config">Config file</strong>

You can find a `config.json` file in the tools directory, which is responsible for customization of the pipeline. You will find 5 entries :

- cache_dir : is the directory where you want to store the dependencies. _(you need to change this once you clone the repo)_
- datasets : a list of the datasets you want to include, feel free to delete some if they are not interesting to you.
- project_path : is the directory where the scripts are located. _(you need to change this once you clone the repo)_
- metrics : is a dictionary of the evaluation metrics used provided by pyserini; the keys are the name of the metrics, while the values are how pyserini refer to them in the results they return _(used with RegEx to retrieve values from output)_
- expansions : is a list of the available expansion methods, that you may use. Feel free to keep only the ones you want to test.

**GROQ API**

GROP API is used to run expansion on their open source models, since it is fast and reliable. Create an account and an API key following this [link](https://console.groq.com/login). Once done, make sure to add the key as an environment variable with the name : `GROQ_KEY`

## The pipeline

The code is prepared such that you only modify the `config.json` file, once you make the necessary changes, you only need to run the `main.py` file.
