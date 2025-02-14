from transformers import pipeline

# cd /tmp/imageia/processor-pyclient/
# python3 llm_simple.py

messages = [
    {"role": "user", "content": "Puedes responder en español?"},
]

#model_name = "meta-llama/Llama-3.1-8B-Instruct"
#model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
model_name = "NousResearch/Hermes-3-Llama-3.2-3B"
pipe = pipeline("text-generation", model=model_name, max_new_tokens=30)
response = pipe(messages)
print(response)