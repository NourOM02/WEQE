from Dataset import Dataset


dbpedia_entity = Dataset('dbpedia-entity', "GROQ_KEY_1")
dbpedia_entity.expand(dbpedia_entity.examples_path, dbpedia_entity.PRF_path)

fever = Dataset('fever', "GROQ_KEY_1")
fever.expand(fever.examples_path, fever.PRF_path)

fiqa = Dataset('fiqa', "GROQ_KEY_1")
fiqa.expand(fiqa.examples_path, fiqa.PRF_path)

nfcorpus = Dataset('nfcorpus', "GROQ_KEY_1")
nfcorpus.expand(nfcorpus.examples_path, nfcorpus.PRF_path)

