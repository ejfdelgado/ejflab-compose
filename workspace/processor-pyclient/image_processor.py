import asyncio
import sys
import json
import os
from base_procesor import BaseProcessor


class ImageProcessor(BaseProcessor):
    async def localConfigure(self):
        pass

    def get_default_arguments(self):
        return {}

    async def process(self, args, default_arguments):
        named_inputs = args['namedInputs']
        method = args['method']
        # print(json.dumps(named_inputs))
        return {}


async def main():
    processor = ImageProcessor()
    await processor.configure(sys.argv)
    await processor.start_communication()

if __name__ == '__main__':
    asyncio.run(main())
