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
from stopwords import stop_words
from time import perf_counter

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

    
    def _concat(self, query_completion):
        return f"{(query_completion[0] + ' ') * 5} + {query_completion[1]}".replace('\n', ' ').replace('\t', ' ').replace('\r', ' ')

    
    
    def _custom_1(self, **kwargs):
        query = kwargs.get('query')
        id = kwargs['id']
        PRF = ujson.load(open(kwargs['prf_path']))
        completion = ""
        i = 0
        for doc in PRF[str(id)]:
            instruction = "Propose a query that could be used to search for the following document:\n"
            instruction += f"Document: {doc}\n"
            i+=1
            completion += self.generate(instruction) + ' '
            if i == 3:
                break
        instruction = ""
        instruction = f"Write a passage that answers the given query: {query}"
        completion += self.generate(instruction)
        return self._concat((query, completion))
    
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
        prf_docs = PRF[str(id)][:1]
        for doc in prf_docs:
            prompt += doc + '\n'
        prompt += f"Query : {query}" + '\n' + "Passage :"
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

    def _paraphrase(self, **kwargs):
        query = kwargs.get('query')
        id = kwargs.get('id')
        PRF = ujson.load(open(kwargs['prf_path']))
        prompt = f"""Generate 3 relevant paraphrased versions of the question '{query}' and provide a 3-line paragraph response for each one.\n\nContext:\n"""
        prf_docs = PRF[str(id)]
        for doc in prf_docs[:min(3, len(prf_docs))]:
            prompt += doc + '\n'

        completion = self.generate(prompt)
        if completion == None:
            return None
        
        expanded_query = (query + ' ' ) * 5 + ' ' + completion
        
        return expanded_query
    
    def _paraphrase_PRF(self, **kwargs):
        """
        query : str
        """
        query = kwargs.get('query')
        id = kwargs.get('id')
        prompt = f"generate 100 paraphrased versions for the following query : {query}\n Context:\n"
        PRF = ujson.load(open(kwargs['prf_path']))
        i = 0
        for doc in PRF[str(id)]:
            prompt += doc + '\n'
            i+=1
            if i == 2:
                break
        completion = self.generate(prompt)
        if completion == None:
            return None
        start_pos = completion.find("1. ")
        completion = completion[start_pos:]
        words = re.findall(r'\b\w+\b', completion)
        words = set([word.lower() for word in words if (not word.isdigit()) and (word not in stop_words)])
        expanded_query = query + ' '.join(words)
        
        return expanded_query
    
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

                for row in queries.iterrows():
                    query = row[1]['query']
                    id = row[1]['id']
                    if not os.path.exists(f"{dependencies}/datasets/queries/{self.name}/{id}_{self.expansions[0]}.txt"):
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
                            with open(f"{dependencies}/datasets/queries/{self.name}/{id}_{self.expansions[0]}.txt", 'w') as file:
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
                queries[approach]['query'].append(f.read().replace('\n', ' '))
        
        for approach in approaches:
            df = pd.DataFrame(queries[approach])
            df.to_csv(f"{dependencies}/datasets/queries/{self.name}_{approach}.tsv",\
                      sep='\t', index=False, header=False)