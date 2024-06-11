from Dataset import Dataset

msmarco = Dataset('msmarco', "GROQ_KEY_2")
msmarco.expand(msmarco.examples_path, msmarco.PRF_path)

