from Dataset import Dataset

hotpotqa = Dataset('hotpotqa', "GROQ_KEY_2")
hotpotqa.expand(hotpotqa.examples_path, hotpotqa.PRF_path)

nq = Dataset('nq', 'GROQ_KEY_2')
nq.expand(nq.examples_path, nq.PRF_path)