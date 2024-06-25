from Dataset import Dataset
import os

# dbpedia_entity = Dataset('', "GROQ_KEY_1")
# dbpedia_entity.expand(dbpedia_entity.examples_path, dbpedia_entity.PRF_path)
# dbpedia_entity.assemble()

# fever = Dataset('fever', "GROQ_KEY_1")
# fever.expand(fever.examples_path, fever.PRF_path)
# fever.assemble()


scidocs = Dataset('webis-touche2020', "GROQ_KEY_1")
# scidocs.expand(scidocs.examples_path, scidocs.PRF_path)
scidocs.assemble()