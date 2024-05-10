import google.generativeai as genai
import os

# Set the API key
genai.configure(api_key=os.environ["GEMINI_API"])

# Model configuration
config = genai.GenerationConfig(candidate_count = 1,\
stop_sequences = None,\
max_output_tokens = None,\
temperature = None,\
top_p = None,\
top_k = None\
)

# Safety settings
safety =[{
            "category": "HARM_CATEGORY_DANGEROUS",
            "threshold": "BLOCK_NONE",
        },\
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_NONE",
        },\
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_NONE",
        },\
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_NONE",
        },\
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_NONE",
        }
    ]

# Create the model
model = genai.GenerativeModel('gemini-pro', safety_settings=safety)

# Define generate function
def generate(prompt):
    # generate answer
    while True:
        try :
            response.text
        except:
            response = model.generate_content(prompt)
        else : break
    return response.text