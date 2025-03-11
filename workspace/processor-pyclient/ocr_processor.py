import asyncio
import sys
import json
import os
from base_procesor import BaseProcessor
from PIL import Image
import io
import pytesseract

class OCRProcessor(BaseProcessor):
    async def localConfigure(self):
        pass

    def get_default_arguments(self):
        return {}

    async def get_text(self, args, default_arguments):
        named_inputs = args['namedInputs']
        byte_array = named_inputs['bytes']
        byte_stream = io.BytesIO(byte_array)
        image_data = Image.open(byte_stream)
        text = pytesseract.image_to_string(image_data)
        return {
            'status': 'ok',
            'text': text,
        }

    async def process(self, args, default_arguments):
        method = args['method']
        if method == "get_text":
            return await self.get_text(args, default_arguments)
        return {}


async def main():
    processor = OCRProcessor()
    

    await processor.configure(sys.argv)
    await processor.start_communication()

if __name__ == '__main__':
    asyncio.run(main())
