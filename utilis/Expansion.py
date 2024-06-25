# Load parameters
import ujson
import os
config = f"{os.path.dirname(os.path.abspath(__file__))}/config.json"
with open(config, 'r') as json_file:
    config = ujson.load(json_file)
    raw_datasets = config['raw_datasets']
    datasets = config['datasets']
    project_path = config['project_path']
    dependencies = config['dependencies']
    expansions = config['expansions']
    metrics = config['metrics']

from groq import Groq
import pandas as pd
import numpy as np
import re
from time import perf_counter
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-MiniLM-L6-v2')
#import nltk
#import time
from nltk.tokenize import sent_tokenize
#nltk.download('punkt')
from tqdm import tqdm


# Logs
if not os.path.exists(f"{dependencies}/logs"):
    os.makedirs(f"{dependencies}/logs")
time_logs = open(f"{dependencies}/logs/time_logs.txt", 'a+', encoding='utf-8')
expansions_logs = open(f"{dependencies}/logs/expansions_logs.txt", 'a+', encoding='utf-8')

class Expansion:
    def __init__(self, dataset, key) -> None:
        self.name = dataset
        self.approach = None
        self._queries = f"{dependencies}/datasets/queries/{self.name}.tsv"
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
        self.queries_path = f"{dependencies}/datasets/queries/{self.name}{suffix}.tsv"
        self.run_path = f"{dependencies}/.runs/{self.name}{suffix}.tsv"

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

    # simple concatenation used by Q2D, Q2D_PRF, Q2D_ZS
    def _concat(self, query_completion):
        return f"{(query_completion[0] + ' ') * 5} + {query_completion[1]}".replace('\n', ' ').replace('\t', ' ').replace('\r', ' ')
    
    def _Q2D(self, **kwargs):
        """
        query : str
        examples_path : str
        """
        query = kwargs['query']
        instruction = f"The following 2 examples will help you understand the format of the passage you will generate\n"
        prompt = instruction + '\n'
        examples = pd.read_csv(kwargs['examples_path'], sep='\t', header=None)
        for i in range(2):
            example = examples.iloc[np.random.randint(0,len(examples))]
            prompt += f"Example{i+1}\nQuery: {example[0]}" +'\n'+ f"Passage : {example[1]}" +'\n\n'
        prompt += "write a passage that answers the following query\n" + f"Query : {query}" + '\n' + "Passage :"
        completion = self.generate(prompt)
        if completion == None:
            return None
        return self._concat((query, completion))
    
    def _Q2D_PRF(self, **kwargs):
        """
        query : str
        id : int
        prf_path : str
        """
        query = kwargs.get('query')
        id = kwargs.get('id')
        PRF = ujson.load(open(kwargs['prf_path']))
        instruction = "Write a passage that answers the given query based on the context:"
        prompt = instruction + '\nContext:\n'
        prf_docs = PRF[str(id)][:min(4, len(PRF[str(id)]))]
        for doc in prf_docs:
            prompt += doc + '\n'
        instruction = f"Query : {query}" + '\n' + "Passage :"
        if len(prompt) > 8192 - len(instruction):
            prompt = prompt[:8192 - len(instruction)]
        completion = self.generate(prompt)
        if completion == None:
            return None
        return self._concat((query, completion))
    
    def _Q2D_ZS(self, **kwargs):
        """
        query : str
        """
        query = kwargs.get('query')
        prompt = f"Write a passage that answers the given query: {query}"
        completion = self.generate(prompt)
        if completion == None:
            return None
        return self._concat((query, completion))

    def _WEQE(self, **kwargs):
        query = kwargs.get('query')
        id = kwargs.get('id')
        examples_path = kwargs.get('examples_path')
        prf_path = kwargs.get('prf_path')


        def generate_prompt(self, query, id, examples_path, prf_path, truncate=False):
            prompt = f"The following 4 examples will help you understand the format of the passage you will generate\n"
            examples = pd.read_csv(examples_path, sep='\t', header=None)
            for i in range(4):
                example = examples.iloc[np.random.randint(0,len(examples))]
                if not truncate:
                    prompt += f"Example{i+1}\nQuery: {example[0]}" +'\n'+ f"Passage : {example[1]}" +'\n\n'
                else:
                    prompt += f"Example{i+1}\nQuery: {example[0][:1800]}" +'\n'+ f"Passage : {example[1][:1800]}" +'\n\n'
            prompt = ""
            prompt += f"""Generate 4 relevant sub-queries that should be searched to answer the following query "{query}" and provide a passage that answers each one of them in a JSON format.\n"""
            prompt += "\nThe output format should be as follows:\n[\n\t{\n\t\t\"query\": sub_query1,\n\t\t\"answer\": answer1\n\t},\n\t{\n\t\t\"query\": sub_query2,\n\t\t\"answer\": answer2\n\t},\n\t{\n\t\t\"query\": sub_query3,\n\t\t\"answer\": answer3\n\t},\n\t{\n\t\t\"query\": sub_query4,\n\t\t\"answer\": answer4\n\t}\n]\nsuch that sub_queryk and answerk are of type string\n"
            if len(prompt) > 8192:
                return generate_prompt(query, id, examples_path, prf_path, truncate=True)
            return prompt

        def generate_completion(self, id, prompt):
            completion_file = f"{dependencies}/datasets/queries/{self.name}_raw/{id}_WEQE.json"
            if os.path.exists(completion_file):
                return 
            completion = self.generate(prompt)
            if completion == None:
                return generate_completion(id, prompt)
            pattern = r'\[\s*{.*?}\s*\]'
            matches = re.search(pattern, completion, re.DOTALL)
            if matches:
                cleaned = re.sub(r'\s*({|}|,|:|\[|\])\s*', r'\1', matches.group(0))
                try:
                    generated_docs = ujson.loads(cleaned)
                    with open(completion_file, 'w') as file:
                        ujson.dump(generated_docs, file, indent=4)
                except:
                    generate_completion(id, prompt)
            else:
                generate_completion(id, prompt)

        def determine_sentences(self, id, query, generated_docs, PRF_docs):
            file = f"{dependencies}/datasets/queries/{self.name}/{id}_P&RDc.txt"
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

        prompt = generate_prompt(query, id, examples_path, prf_path)
        generate_completion(id, prompt)
        generated_docs = ujson.load(open(f"{dependencies}/datasets/queries/{self.name}_raw/{id}_WEQE.json"))
        try:
            generated_docs = [f"{doc['query']}.{doc['answer']}" for doc in generated_docs]
        except:
            os.remove(f"{dependencies}/datasets/queries/{self.name}_raw/{id}_WEQE.json")
            return self._WEQE(query=query, id=id, examples_path=examples_path, prf_path=prf_path, dataset=self.name)
        PRF_docs = ujson.load(open(prf_path))
        PRF_docs = PRF_docs[str(id)][:min(3, len(PRF_docs[str(id)]))]
        if PRF_docs == []:
            PRF_docs.append("The query is not relevant to any document in the dataset.")
        ordered_sentences = self.determine_sentences(id, query, generated_docs, PRF_docs)
        return True

    def expand(self, examples_path, prf_path):
        if self.expansions == []:
            return
        else:
            self.approach = self.expansions[0]
            self.init_paths()
            if not os.path.exists(self.queries_path):
                if not os.path.exists(f"{dependencies}/datasets/queries/{self.name}"):
                    os.makedirs(f"{dependencies}/datasets/queries/{self.name}")
                scope = getattr(__import__(__name__), "Expansion")(self.name, self.key)
                self.approach = getattr(scope, f"_{self.approach}")
                
                queries = pd.read_csv(self._queries, sep='\t', header=None)
                queries.columns = ['id', 'query']
                extension = 'json' if self.expansions[0]=='WEQE' else 'txt'
                for row in queries.iterrows():
                    query = row[1]['query']
                    id = row[1]['id']
                    if not os.path.exists(f"{dependencies}/datasets/queries/{self.name}/{id}_{self.expansions[0]}.{extension}"):
                        expansions_logs.write(f"{self.name}, {id}, {self.expansions[0]}")
                        time_logs.write(f"{self.name}, {id}, {self.expansions[0]}")
                        status = ""
                        start = perf_counter()
                        while True:
                            expanded = self.approach(query=query, id=id, examples_path=examples_path,\
                                    prf_path=prf_path, dataset=self.name)
                            print(f"Expanding {id}_{self.expansions[0]}", end=' ', flush=True)
                            if expanded == None or len(expanded) <= len(query)*5 + 100:
                                print("Failed...")
                                status += "X"
                                continue
                            with open(f"{dependencies}/datasets/queries/{self.name}/{id}_{self.expansions[0]}.{extension}", 'w') as file:
                                print("Success...")
                                status += "O"
                                file.write(expanded)
                                expansions_logs.write(f", {status}\n")
                                time_logs.write(f", {perf_counter() - start}\n")
                                break
                        

            self.expanded.append(self.expansions[0])
            self.expansions.pop(0)
            self.expand(examples_path, prf_path)

    def assemble(self):
        folder = f"{dependencies}/datasets/queries/{self.name}"
        files = os.listdir(folder)
        approaches = []
        queries = {}
        for file in files:
            approach = file.split('_', 1)[1].split('.')[0]
            if approach not in approaches:
                approaches.append(approach)
                queries[approach] = {
                    'query-id': [],
                    'query' : []
                }
            with open(f"{folder}/{file}", 'r') as f:
                queries[approach]['query-id'].append(file.split('_')[0])
                if file.endswith("json"):
                    for doc in ujson.load(f):
                        queries[approach]['query'].append(f"{doc['query']}.{doc['answer']}")
                else:
                    queries[approach]['query'].append(f.read().replace('\n', ' '))
        
        for approach in approaches:
            df = pd.DataFrame(queries[approach])
            df.to_csv(f"{dependencies}/datasets/queries/{self.name}_{approach}.tsv",\
                      sep='\t', index=False, header=False)

    def offline_expand(self, approach, prf_path):
        self.approach = approach
        self.init_paths()
        original_queries = pd.read_csv(self._queries, sep='\t', header=None)
        expanded_queries = pd.read_csv(self.queries_path, sep='\t', header=None)
        PRF = ujson.load(open(prf_path))
        emb_query = {
            '5_4':{
                'query-id': [],
                'query' : []
            },
            '5_6':{
                'query-id': [],
                'query' : []
            },
            '5_8':{
                'query-id': [],
                'query' : []
            },
            '3_4':{
                'query-id': [],
                'query' : []
            },
            '3_6':{
                'query-id': [],
                'query' : []
            },
            '3_8':{
                'query-id': [],
                'query' : []
            },
            '2_4':{
                'query-id': [],
                'query' : []
            },
            '2_6':{
                'query-id': [],
                'query' : []
            },
            '2_8':{
                'query-id': [],
                'query' : []
            },
            '1_4':{
                'query-id': [],
                'query' : []
            },
            '1_6':{
                'query-id': [],
                'query' : []
            },
            '1_8':{
                'query-id': [],
                'query' : []
            }
        }

        for row in tqdm(original_queries.iterrows()):
            query = row[1][1]
            id = row[1][0]
            prf_docs:list = PRF[str(id)]
            expanded_query = expanded_queries[expanded_queries[0] == id][1].values[0]
            expanded_query = expanded_query[(len(query) + 1)*5:]
            if len(prf_docs) == 0:
                prf_docs.append("The query is not relevant to any document in the dataset.")
            sentences = [sentence for sentence in sent_tokenize(expanded_query + ' ' + prf_docs[0])]
            query_embedding = model.encode(query, convert_to_tensor=True)
            prf_embedding = model.encode(prf_docs, convert_to_tensor=True) #
            gen_embedding = model.encode([expanded_query], convert_to_tensor=True) #
            sentence_embeddings = model.encode(sentences, convert_to_tensor=True)
            #cosine_scores = util.pytorch_cos_sim(query_embedding, sentence_embeddings)[0]
            cosine_scores_q = util.pytorch_cos_sim(query_embedding, sentence_embeddings)[0] #
            cosine_scores_prf = util.pytorch_cos_sim(prf_embedding, sentence_embeddings)[0] #
            cosine_scores_gen = util.pytorch_cos_sim(gen_embedding, sentence_embeddings)[0] #
            weight_q = 0.4 #
            weight_prf = 0.3 #
            weight_gen = 0.3 #
            cosine_scores = weight_q * cosine_scores_q + weight_prf * cosine_scores_prf + weight_gen * cosine_scores_gen #
                
            relevant_indices = np.argsort(-cosine_scores)
            relevant_sentences = [sentences[i] for i in relevant_indices]
            for key in emb_query.keys():
                n_sentences = int(key.split('_')[1])
                n_repetitions = int(key.split('_')[0])
                expanded_query = ' '.join(relevant_sentences[:n_sentences])
                emb_query[key]['query-id'].append(id)
                emb_query[key]['query'].append((query + ' ' ) * n_repetitions + expanded_query)
        for key in emb_query.keys():
            df = pd.DataFrame(emb_query[key])
            df.to_csv(f"{dependencies}/datasets/queries/{self.name}_{approach}_{key}.tsv", sep='\t', index=False, header=False)