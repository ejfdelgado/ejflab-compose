import asyncio
import sys
import json
import os
from base_procesor import BaseProcessor
import importlib.resources as importlib_resources
from gpt4all import GPT4All

# LlaVa
# downloads / loads a 4.66GB LLM
device = "cpu"
if os.environ['DEVICE'] == "cuda":
    gpus = GPT4All.list_gpus()
    if len(gpus) > 0:
        device = gpus[0]

print(f"device... {device}")

# gpt4all-13b-snoozy-q4_0.gguf
# Meta-Llama-3-8B-Instruct.Q4_0.gguf
# orca-mini-3b-gguf2-q4_0.gguf (es algo de salud)
model = "Meta-Llama-3-8B-Instruct.Q4_0.gguf"
if "MODEL" in os.environ:
    model = os.environ["MODEL"]

# model_path=~/.cache/gpt4all/
model = GPT4All(model, device=device, allow_download=True)
print("Model loaded...")

# print(repr(model.config['systemPrompt']))
# print(repr(model.config['promptTemplate']))

# /root/.cache/gpt4all/Meta-Llama-3-8B-Instruct.Q4_0.gguf
# cp /usr/local/lib/python3.10/dist-packages/gpt4all/gpt4all.py /tmp/imageia/processor-pyclient/gpt4all.py

class LLMProcessor(BaseProcessor):
    async def localConfigure(self):
        pass

    def get_default_arguments(self):
        return {
            'maxTokens': 1000,
            'systemMessage': 'Eres un asistente en español',
            'chatTemplate': "### Human:\n{0}\n\n### Assistant:\n",
            'prefixPrompt': 'En español describe la conversación: ',
            'streaming': False
        }

    def construct_prompt(self, current_chat_session):
        """Construct the prompt from the current session."""
        prompt = ""
        for message in current_chat_session:
            prompt += f"{message['role']}: {message['content']}\n"
        return prompt

    async def chat(self, args, default_arguments):
        named_inputs = args['namedInputs']
        response = ''
        session = None
        if 'session' in named_inputs:
            session = named_inputs['session']
        system_message = default_arguments['systemMessage']
        chat_template = default_arguments['chatTemplate']
        
        def stream_callback(token: str):
            print(token, end="", flush=True)
        
        with model.chat_session(system_message, chat_template):
            current_prompt = named_inputs['message']
            if session is not None:
                session.append({"role": "user", "content": named_inputs['message']})
                current_prompt = model._format_chat_prompt_template(session)
            response = model.generate(current_prompt, 
                                      max_tokens=default_arguments['maxTokens'], 
                                      streaming=default_arguments['streaming'],
                                      #callback=stream_callback
                                      )
            session = model.current_chat_session
        if default_arguments['streaming']:
            return response
        return {
            'response': response,
            'session': session,
            'current_prompt': current_prompt
        }

    async def train(self, args, default_arguments):
        named_inputs = args['namedInputs']
        return {}
    
    async def summary(self, args, default_arguments):
        named_inputs = args['namedInputs']
        source = named_inputs['source']
        rows = source['rows']
        #print(default_arguments)
        my_query = default_arguments['prefixPrompt']
        for row in rows:
            my_query = my_query + row["txt"] + "\n"
        system_template = ''
        prompt_template = '<|start_header_id|>user<|end_header_id|>\n\n{0}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{1}<|eot_id|>'
        with model.chat_session(system_template, prompt_template):
            response = (model.generate(
                my_query, max_tokens=default_arguments['maxTokens']))
            session = model.current_chat_session
        #print(response)
        return {
            "summary": response,
            #"session": session,
            #"query": my_query
        }

    async def process(self, args, default_arguments):
        method = args['method']
        if method == "chat":
            return await self.chat(args, default_arguments)
        if method == "summary":
            return await self.summary(args, default_arguments)
        if method == "train":
            return await self.train(args, default_arguments)
        return {}


async def main():
    processor = LLMProcessor()
    await processor.configure(sys.argv)
    await processor.start_communication()

if __name__ == '__main__':
    asyncio.run(main())
