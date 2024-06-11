from Dataset import Dataset

cqadupstack = Dataset('webis-touche2020', "GROQ_KEY_1")

print(cqadupstack.evaluate())
cqadupstack.approach = "Q2D"
cqadupstack.init_paths()
cqadupstack.retrieve()
print(cqadupstack.evaluate())

cqadupstack.queries_path = "/home/application/.QueryExpansionLLMs/datasets/queries/webis-touche2020_paraphrase.tsv"
cqadupstack.retrieve()
print(cqadupstack.evaluate())

cqadupstack.queries_path = "/home/application/.QueryExpansionLLMs/datasets/queries/webis-touche2020_paraphrase_x.tsv"
cqadupstack.retrieve()
print(cqadupstack.evaluate())