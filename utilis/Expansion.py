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

from pyserini.search.lucene import LuceneSearcher
from groq import Groq
import pandas as pd
import numpy as np
from time import sleep

class Expansion:
    def __init__(self, dataset) -> None:
        self.name = dataset
        self.approach = None
        self._queries = f"{dependencies}/datasets/queries/{self.name}.tsv"
        self.queries_path = ""
        self.run_path = ""
        self.client = self._API()
        self.init_paths()
        self.expansions = expansions
        self.expanded = []
        
    def init_paths(self):
        suffix = f"_{self.approach}" if self.approach else ""
        self.queries_path = f"{dependencies}/datasets/queries/{self.name}{suffix}.tsv"
        self.run_path = f"{dependencies}/.runs/{self.name}{suffix}.tsv"

    def _API(self):
        client = Groq(
            api_key=os.environ.get("GROQ_KEY_2"),
        )
        return client
    
    def generate(self, prompt, model="llama3-8b-8192"):
        chat_completion = self.client.chat.completions.create(
        messages=[
                {
                    "role": "user",
                    "content": f"{prompt}",
                }
            ],
            model=model,
        )
        return chat_completion.choices[0].message.content
    
    def _concat(self, query_completion):
        return f"{query_completion[0] * 5} + {query_completion[1]}".replace('\n', ' ')

    def _Q2D(self, **kwargs):
        """
        query : str
        examples_path : str
        """
        query = kwargs['query']
        examples_path = kwargs['examples_path']
        print("*", sep='', end='', flush=True)
        instruction = f"Write a passage that answers the given query.\n\
        Here are examples of random queries (that aren't necessarly relevant) and the passages that answer them\n"
        prompt = instruction
        examples = pd.read_csv(examples_path, sep='\t', header=None)
        for i in range(4):
            example = examples.iloc[np.random.randint(0,len(examples))]
            prompt += f"Query: {example[0]}" +'\n'+ f"Passage : {example[1]}" +'\n\n'
        
        prompt += instruction + f"Query : {query}" + '\n' + "Passage :"
        completion = self.generate(prompt)
        sleep(1)
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
        print("*", sep='', end='', flush=True)
        instruction = "Write a passage that answers the given query based on the context:"
        prompt = instruction + '\nContext:\n'
        prf_docs = PRF[str(id)]
        for doc in prf_docs:
            prompt += doc + '\n'
        prompt += f"Query : {query}" + '\n' + "Passage :"
        completion = self.generate(prompt)
        sleep(1)
        return self._concat((query, completion))
    
    def _Q2D_ZS(self, **kwargs):
        """
        query : str
        """
        query = kwargs.get('query')
        print("*", sep='', end='', flush=True)
        prompt = f"Write a passage that answers the given query: {query}"
        completion = self.generate(prompt)
        sleep(1)
        return self._concat((query, completion))
    
    def expand(self, examples_path, prf_path):
        if self.expansions == []:
            return
        else:
            self.approach = self.expansions[0]
            self.init_paths()
            if not os.path.exists(self.queries_path):
                scope = getattr(__import__(__name__), "Expansion")(self.name)
                self.approach = getattr(scope, f"_{self.approach}")
                
                queries = pd.read_csv(self._queries, sep='\t', header=None)
                queries.columns = ['id', 'query']

                queries["query"] = queries.apply(lambda row: self.approach(query=
                    row['query'], id=row['id'], prf_path=prf_path, examples_path=
                    examples_path, dataset=self.name), axis=1)

                queries.to_csv(self.queries_path, sep='\t', index=False, header=None)

            self.expanded.append(self.expansions[0])
            self.expansions.pop(0)
            self.expand(examples_path, prf_path)