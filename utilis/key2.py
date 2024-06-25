from Dataset import Dataset

fiqa = Dataset('fiqa', "GROQ_KEY_2")
fiqa.expand(fiqa.examples_path, fiqa.PRF_path)  

hotpotqa = Dataset('hotpotqa', "GROQ_KEY_2")
hotpotqa.expand(hotpotqa.examples_path, hotpotqa.PRF_path)

msmarco = Dataset('msmarco', "GROQ_KEY_2")
msmarco.expand(msmarco.examples_path, msmarco.PRF_path)

nfcorpus = Dataset('nfcorpus', "GROQ_KEY_2")
nfcorpus.expand(nfcorpus.examples_path, nfcorpus.PRF_path)

nq = Dataset('nq', "GROQ_KEY_2")
nq.expand(nq.examples_path, nq.PRF_path)

quora = Dataset('quora', "GROQ_KEY_2")
quora.expand(quora.examples_path, quora.PRF_path)
