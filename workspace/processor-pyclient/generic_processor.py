import asyncio
import sys
import json
import os
from base_procesor import BaseProcessor


class GenericProcessor(BaseProcessor):
    async def localConfigure(self):
        pass

    def get_default_arguments(self):
        return {}

    async def custom(self, args, default_arguments):
        named_inputs = args['namedInputs']
        print(json.dumps(named_inputs))
        return {
            'status': 'ok'
        }

    async def process(self, args, default_arguments):
        method = args['method']
        if method == "custom":
            return await self.custom(args, default_arguments)
        return {}


async def main():
    processor = GenericProcessor()
    await processor.configure(sys.argv)
    await processor.start_communication()

if __name__ == '__main__':
    asyncio.run(main())
