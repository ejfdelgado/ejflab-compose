from datasets import Dataset
from transformers import PreTrainedTokenizerFast, LlamaForCausalLM

# cd /tmp/imageia/processor-pyclient/
# python3 llm_hermes.py

data = [
    {"instruction": "What is the capital of France?", "input": "", "output": "Paris"},
    {"instruction": "Explain the theory of relativity.", "input": "", "output": "Einstein's theory states..."}
]
dataset = Dataset.from_list(data)

model_name = "NousResearch/Hermes-3-Llama-3.2-3B"
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name, device_map="auto", load_in_8bit=True)
