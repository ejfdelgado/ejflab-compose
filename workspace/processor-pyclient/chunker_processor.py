import asyncio
import sys
import json
import os
from base_procesor import BaseProcessor
import semchunk
from transformers import AutoTokenizer

class ChunkerProcessor(BaseProcessor):
    async def localConfigure(self):
        pass

    def get_default_arguments(self):
        return {
            'chunkSize': 4
        }
    
    async def split(self, args, default_arguments):
        named_inputs = args['namedInputs']
        text = named_inputs['text']
        chunk_size = default_arguments['chunkSize']
        print(f"chunk_size={chunk_size}")
        chunker = semchunk.chunkerify(AutoTokenizer.from_pretrained('isaacus/emubert'), chunk_size)
        response = chunker(text)
        return {
            'chunks': response
        }

    async def process(self, args, default_arguments):
        
        method = args['method']
        if method == "split":
            return await self.split(args, default_arguments)
        return {}


async def main():
    processor = ChunkerProcessor()
    await processor.configure(sys.argv)
    await processor.start_communication()

if __name__ == '__main__':
    asyncio.run(main())
