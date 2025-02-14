import asyncio
import sys
from base_procesor import BaseProcessor


class FaceDetectionProcessor(BaseProcessor):
    async def localConfigure(self):
        pass

    def get_default_arguments(self):
        return {'presition': 0.8}

    async def process(self, args, default_arguments):
        named_inputs = args['namedInputs']
        method = args['method']
        return {'boundingBoxes': []}


async def main():
    processor = FaceDetectionProcessor()
    await processor.configure(sys.argv)
    await processor.start_communication()

if __name__ == '__main__':
    asyncio.run(main())
