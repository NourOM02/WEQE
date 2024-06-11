from Dataset import Dataset


msmarco = Dataset('msmarco', "GROQ_KEY_1")
msmarco.expand(msmarco.examples_path, msmarco.PRF_path)

quora = Dataset('quora', "GROQ_KEY_1")
quora.expand(quora.examples_path, quora.PRF_path)
