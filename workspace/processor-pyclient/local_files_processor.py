import asyncio
import sys
from base_procesor import BaseProcessor
from random import randint
from time import sleep
import time
import os

# ffmpeg -err_detect ignore_err -i video001.mp4 -c copy video001_.mp4
# ffmpeg -ss 00:19:06 -to 00:22:37 -i 16ef076a-2024_10_24-14_33_36-TAGEDX3-VIDEO-VOLUNTARY.mp4 -c copy 16ef076a-2024_10_24-14_33_36-TAGEDX3-VIDEO-VOLUNTARY_.mp4

class LocalFilesProcessor(BaseProcessor):
    async def localConfigure(self):
        pass

    def get_default_arguments(self):
        return {
            'extension': "mp3",
            'preserve_temp_files': False,
        }

    async def process(self, args, default_arguments):
        named_inputs = args['namedInputs']
        method = args['method']
        inputs = []
        if 'inputs' in args:
            inputs = args['inputs']

        if (method == "read"):
            # Read custom inputs
            timeline = named_inputs['timeline']
            source = named_inputs['source']
            bytes = await self.read(timeline, source, default_arguments, inputs)
            return {'bytes': bytes}
        elif (method == "echo"):
            return named_inputs
        elif (method == "write"):
            print("Write...")
            bytes = named_inputs['bytes']
            data_type = str(type(bytes))

            if (data_type == "<class 'bytes'>"):
                print("Storing bytes")
                self.store_bytes_locally(
                    bytes, default_arguments["extension"], args)
            elif (data_type == "<class 'list'>"):
                print("Storing bytes array")
                id = args['id']
                count = 0
                for one_bytes in bytes:
                    args['id'] = f"{id}_{count}"
                    self.store_bytes_locally(
                        one_bytes, default_arguments["extension"], args)
                    count = count + 1
            else:
                raise Exception(f"data_type {data_type} not valid")
            print("Write... OK!")
            return {
                "time": 1000*int(time.time())
            }
        elif (method == "metavideo"):
            print("metavideo...")
            source = named_inputs['source']
            media = named_inputs['media']
            metadata = await self.get_video_attributes(source, media)
            return {
                "metadata": metadata
            }
        elif (method == "bytesmeta"):
            bytes = named_inputs['bytes']
            media = named_inputs['media']
            path = self.store_bytes_locally(
                bytes, default_arguments["extension"], args)
            metadata = await self.get_video_attributes({'src': path}, media)
            if not default_arguments['preserve_temp_files']:
                os.remove(path)
            return {
                "metadata": metadata
            }
        elif (method == "randomerror"):
            detail = named_inputs['detail']
            scope = named_inputs['scope']
            start = scope['start']
            epoch_time = 1000*int(time.time())
            elapsed_time = (epoch_time - start)/1000
            min = 1
            max = 2
            fail = True
            current = detail['current']
            steps = detail[current]
            current_step = steps[-1]
            max_sum = 0
            for step in steps:
                local_max = max
                if 'max' in current_step:
                    local_max = current_step['max']
                max_sum = max_sum + local_max
                if elapsed_time < max_sum:
                    current_step = step
                    break

            print(f"Using step {current_step}")
            if current_step is not None:
                if 'min' in current_step:
                    min = current_step['min']
                if 'max' in current_step:
                    max = current_step['max']
                if 'fail' in current_step:
                    fail = current_step['fail']
            randval = randint(min, max)
            fail_min = fail/2
            fail_max = 50 + fail_min
            rand_fail = randint(fail_min, fail_max)
            will_fail = rand_fail >= 50

            if will_fail:
                print(f"Fail in {randval} seconds ...")
            else:
                print(f"Success in {randval} seconds ...")
            sleep(randval)

            if will_fail:
                raise Exception(
                    f"Intentional error {rand_fail} at {randval} seconds")
            else:
                return {"detail": detail}

        return {}


async def main():
    processor = LocalFilesProcessor()
    await processor.configure(sys.argv)
    await processor.start_communication()

if __name__ == '__main__':
    asyncio.run(main())
