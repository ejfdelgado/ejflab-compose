import asyncio
import sys
import json
import os
from base_procesor import BaseProcessor
import subprocess
import base64
import uuid
import wave

# It defines which model to use given language
LANGUAGE_MAP = {
    'es': 'es_MX-claude-high.onnx',
    'en': 'en_US-hfc_female-medium.onnx',
}

class Text2SpeechProcessor(BaseProcessor):
    async def localConfigure(self):
        pass

    def get_default_arguments(self):
        return {
            # en_US-hfc_female-medium.onnx es_MX-claude-high.onnx
            'language': 'es',
            'voice_folder': '/tmp/imageia/processor-pyclient/voices/'
        }
    
    def read_binary_file(self, file_path):
        with open(file_path, 'rb') as file:
            binary_data = file.read()
        return binary_data

    def read_wav_file(self, file_path):
        with wave.open(file_path, 'rb') as wav_file:
            # Get the WAV file parameters
            num_channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            num_frames = wav_file.getnframes()

            # Read all audio frames (raw binary data)
            raw_audio_data = wav_file.readframes(num_frames)

            return {
                'raw_audio_data': raw_audio_data
            }

    async def convert(self, args, default_arguments):
        global LANGUAGE_MAP
        named_inputs = args['namedInputs']
        text = named_inputs['text'].replace("'", "\'")
        language = default_arguments['language']
        model = LANGUAGE_MAP[language]
        voice_folder = default_arguments['voice_folder']

        temp_file_name = f"./temp/{uuid.uuid4()}.wav"
        command = f"echo '{text}' | piper --model {voice_folder}{model} --output_file {temp_file_name}"
        pipe = subprocess.run(["bash", "-c", command],
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE,
                              text=False,
                              bufsize=10**8)
        stdout = pipe.stdout
        stderr = pipe.stderr
        print(stderr)
        #print(stdout)
        
        #audio_readed = self.read_wav_file(temp_file_name)
        #encoded=base64.b64encode(audio_readed['raw_audio_data'])
        audio_readed = self.read_binary_file(temp_file_name)
        encoded=base64.b64encode(audio_readed)

        # delete the file
        os.remove(temp_file_name)

        return {
            'status': 'ok',
            'audio': 'data:audio/wav;base64,'+encoded.decode('utf-8')
        }

    async def process(self, args, default_arguments):
        method = args['method']
        if method == "convert":
            return await self.convert(args, default_arguments)
        return {}


async def main():
    processor = Text2SpeechProcessor()
    await processor.configure(sys.argv)
    await processor.start_communication()

if __name__ == '__main__':
    asyncio.run(main())
