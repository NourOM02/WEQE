# Load parameters
import ujson
import os
from groq import Groq
import pandas as pd
import numpy as np
import re
from time import perf_counter
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-MiniLM-L6-v2')
#import nltk
import time
from nltk.tokenize import sent_tokenize
#nltk.download('punkt')
from tqdm import tqdm
config = f"{os.path.dirname(os.path.abspath(__file__))}/config.json"
with open(config, 'r') as json_file:
    config = ujson.load(json_file)
    datasets = config['datasets']
    project_path = config['project_path']
    cache_dir = config['cache_dir']
    expansions = config['expansions']
    metrics = config['metrics']


# Logs
time_logs = open(f"{cache_dir}/logs/time_logs.txt", 'a+', encoding='utf-8')
expansions_logs = open(f"{cache_dir}/logs/expansions_logs.txt", 'a+',\
                       encoding='utf-8')

class Expansion:
    def __init__(self, dataset, key) -> None:
        self.name = dataset
        self.approach = None
        self._queries = f"{cache_dir}/datasets/queries/{self.name}.tsv"
        self.queries_path = ""
        self.run_path = ""
        self.client = ""
        self.model = "llama3-8b-8192"
        self.key = key
        self.init_paths()
        self.expansions = expansions
        self.expanded = []
        
    def init_paths(self):
        suffix = f"_{self.approach}" if self.approach else ""
        q = f"{self.name}{suffix}.tsv"
        self.queries_path = f"{cache_dir}/datasets/queries/{q}"
        self.run_path = f"{cache_dir}/.runs/{self.name}{suffix}.tsv"

    def _API(self, key):
        client = Groq(
            api_key=os.environ.get(key),
        )
        return client
    
    def generate(self, prompt):

        def completion(self, prompt):
            chat_completion = self.client.chat.completions.create(
                messages=[
                        {
                            "role": "user",
                            "content": f"{prompt}",
                        }
                    ],
                    model=self.model,
                )
            return chat_completion.choices[0].message.content
        
        try:
            self.client = self._API(self.key)
            return completion(self, prompt)
        except Exception as e:
            print(e)
    
    def generate_prompt(self, query, id, examples_path, prf_path, approach, truncate=False):
        if approach == 'Q2D':
            prompt = f"The following 4 examples will help you understand the format\
of the passage you will generate\n\n"
            examples = pd.read_csv(examples_path, sep='\t', header=None)
            to_add = [examples.iloc[np.random.randint(0,len(examples))]
                      for _ in range(4)]
            if truncate:
                to_add = [f"Query: {addition[0]}\nPassage: {addition[1][:4400]}\n\n"
                        for addition in to_add]
            else:
                to_add = [f"Query: {addition[0]}\nPassage: {addition[1]}\n\n"
                        for addition in to_add]
            instruction = "write a passage that answers the following query\n" +\
                            f"Query : {query}" + '\n' + "Passage :"
            suffix = ""
        elif approach == 'Q2D_PRF':
            prompt = "Write a passage that answers the given query based on \
the context:\n\n"
            PRF = ujson.load(open(prf_path))
            to_add = PRF[str(id)][:min(3, len(PRF[str(id)]))]
            if truncate:
                to_add = [f"{doc[:4500]}\n" for doc in to_add]
            else:
                to_add = [f"{doc}\n" for doc in to_add]
            instruction = f"\nQuery : {query}" + '\n' + "Passage :"
            suffix = ""
        elif approach == 'Q2D_ZS':
            prompt = ""
            to_add = []
            instruction = f"Write a passage that answers the given query: {query}"
            suffix = ""
        elif approach == 'WEQE':
            prompt = f"The following 4 examples will help you understand the format\
of the passage you will generate\n\n"
            examples = pd.read_csv(examples_path, sep='\t', header=None)
            to_add = [examples.iloc[np.random.randint(0,len(examples))]
                      for _ in range(4)]
            if truncate:
                to_add = [f"Query: {addition[0]}\nPassage: {addition[1][:4400]}\n\n"
                        for addition in to_add]
            else:
                to_add = [f"Query: {addition[0]}\nPassage: {addition[1]}\n\n"
                        for addition in to_add]
            instruction = "Generate 4 relevant sub-queries that should be searched"+\
            f" to answer the following query \"{query}\" and provide a passage that"+\
            " answers each one of them in a JSON format."
            suffix = "\nThe output format should be as follows:\n[\n\t{\n\t\t\"" + \
            "query\": sub_query1,\n\t\t\"answer\": answer1\n\t},\n\t{\n\t\t\"query\""\
            + ": sub_query2,\n\t\t\"answer\": answer2\n\t},\n\t{\n\t\t\"query\": "+\
            "sub_query3,\n\t\t\"answer\": answer3\n\t},\n\t{\n\t\t\"query\": " + \
            "sub_query4,\n\t\t\"answer\": answer4\n\t}\n]\nsuch that sub_queryk" + \
            " and answerk are of type string\n"
            
        for addition in to_add:
            prompt += addition
        prompt += instruction
        prompt += suffix

        return prompt
    
    def generate_completion(self, id, prompt, approach):
        ext = 'json' if approach == 'WEQE' else 'txt'
        fd = '_raw' if approach == 'WEQE' else '' 
        completion_file = f"{cache_dir}/datasets/queries/{self.name}{fd}/{id}_{approach}.{ext}"
        if os.path.exists(completion_file):
            return
        completion = self.generate(prompt)
        if completion == None or len(completion) <= 100:
            expansions_logs.write("X")
            return self.generate_completion(id, prompt, approach)
        if approach == 'WEQE':
            pattern = r'\[\s*{.*?}\s*\]'
            matches = re.search(pattern, completion, re.DOTALL)
            if matches:
                cleaned = re.sub(r'\s*({|}|,|:|\[|\])\s*', r'\1', matches.group(0))
                try:
                    generated_docs = ujson.loads(cleaned)
                    with open(completion_file, 'w') as file:
                        ujson.dump(generated_docs, file, indent=4)
                        expansions_logs.write(f"O\n")
                except:
                    expansions_logs.write(f"X")
                    self.generate_completion(id, prompt, approach)
            else:
                expansions_logs.write(f"X")
                self.generate_completion(id, prompt, approach)
        else:
            with open(completion_file, 'w') as file:
                file.write(completion)
                expansions_logs.write(f"O\n")

    def determine_sentences(self, id, query, generated_docs, PRF_docs):
        file = f"{cache_dir}/datasets/queries/{self.name}/{id}_WEQE.txt"
        # Tokenize documents into sentences
        generated_sentences = [sentence for doc in generated_docs for sentence in sent_tokenize(doc)]
        prf_sentences = [sentence for doc in PRF_docs for sentence in sent_tokenize(doc)]
        # Merge all sentences
        sentences = generated_sentences + prf_sentences
        
        # Calculte embeddings
        query_embedding = model.encode(query, convert_to_tensor=True)
        prf_embedding = model.encode(prf_sentences, convert_to_tensor=True)
        gen_embedding = model.encode(generated_sentences, convert_to_tensor=True)
        sentence_embeddings = model.encode(sentences, convert_to_tensor=True)

        # Calculate cosine similarity
        cosine_scores_q = util.pytorch_cos_sim(query_embedding, sentence_embeddings)[0]
        cosine_scores_prf = util.pytorch_cos_sim(prf_embedding, sentence_embeddings)[0]
        cosine_scores_gen = util.pytorch_cos_sim(gen_embedding, sentence_embeddings)[0]

        # Weights
        weight_q = 0.4
        weight_prf = 0.3
        weight_gen = 0.3

        # Calculate final cosine similarity
        cosine_scores = weight_q * cosine_scores_q + weight_prf * cosine_scores_prf + weight_gen * cosine_scores_gen

        # Sort sentences by cosine similarity
        relevant_indices = np.argsort(-cosine_scores)
        relevant_sentences = [sentences[i] for i in relevant_indices]

        # Write to file
        with open(file, 'w') as f:
            for sentence in relevant_sentences:
                f.write(f"{sentence}\n")

        return relevant_sentences

    def expand(self, examples_path, prf_path):
        for expansion in self.expansions:
            self.approach = expansion
            self.init_paths()
            if not os.path.exists(self.queries_path):
                if not os.path.exists(f"{cache_dir}/datasets/queries/{self.name}"):
                    os.makedirs(f"{cache_dir}/datasets/queries/{self.name}")
                if not os.path.exists(f"{cache_dir}/datasets/queries/{self.name}_raw"):
                    os.makedirs(f"{cache_dir}/datasets/queries/{self.name}_raw")
                queries = pd.read_csv(self._queries, sep='\t', header=None)
                queries.columns = ['id', 'query']
                
                for row in queries.iterrows():
                    query = row[1]['query']
                    id = row[1]['id']
                    if not os.path.exists(f"{cache_dir}/datasets/queries/{self.name}/{id}_{self.expansions[0]}.txt"):
                        expansions_logs.write(f"{self.name}, {id}, {self.expansions[0]}, ")
                        time_logs.write(f"{self.name}, {id}, {self.expansions[0]}")
                        print(f"Expanding {self.name} {id}_{self.expansions[0]}...", end=' ', flush=True)
                        start = perf_counter()
                        prompt = self.generate_prompt(query=query, id=id, examples_path=examples_path,\
                                                      prf_path=prf_path, approach=self.approach)
                        self.generate_completion(id=id, prompt=prompt, approach=self.approach)

                        if self.approach == "WEQE":
                            try:
                                generated_docs = ujson.load(open(f"{cache_dir}/datasets/queries/{self.name}_raw/{id}_WEQE.json"))
                                generated_docs = [f"{doc['query']}.{doc['answer']}" for doc in generated_docs]
                            except:
                                os.remove(f"{cache_dir}/datasets/queries/{self.name}_raw/{id}_WEQE.json")
                                return self.generate_completion(id=id, prompt=prompt, approach=self.approach)
                            PRF_docs = ujson.load(open(prf_path))
                            PRF_docs = PRF_docs[str(id)][:min(3, len(PRF_docs[str(id)]))]
                            if PRF_docs == []:
                                PRF_docs.append("The query is not relevant to any document in the dataset.")
                            self.determine_sentences(id, query, generated_docs, PRF_docs)
                        
                        print("Success...")
                        time_logs.write(f", {perf_counter() - start}\n")

            self.expanded.append(expansion)

    def assemble(self, q = 5, s = 8):
        folder = f"{cache_dir}/datasets/queries/{self.name}"
        files = os.listdir(folder)
        approaches = []
        queries = {}
        org_queries = pd.read_csv(self._queries, sep='\t', header=None)
        for file in files:
            approach = file.split('_', 1)[1].split('.')[0]
            if approach not in approaches:
                approaches.append(approach)
                queries[approach] = {
                    'query-id': [],
                    'query' : []
                }
            with open(f"{folder}/{file}", 'r') as f:
                id_ = file.split('_')[0]
                q_org = org_queries[org_queries[0] == int(id_)][1].values[0]
                queries[approach]['query-id'].append(id_)
                if approach != 'WEQE':
                    q_prime = (q_org + " ") * q + f.read().replace('\n', ' ')
                    queries[approach]['query'].append(q_prime)
                else:
                    sentences = f.readlines()
                    q_prime = (q_org + " ") * q + " ".join(sentences[:s])
                    queries[approach]['query'].append(q_prime.replace('\n', ' '))

        
        for approach in approaches:
            df = pd.DataFrame(queries[approach])
            df.to_csv(f"{cache_dir}/datasets/queries/{self.name}_{approach}.tsv",\
                      sep='\t', index=False, header=False)