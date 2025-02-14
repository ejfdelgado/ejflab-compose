import asyncio
import sys
import os
from base_procesor import BaseProcessor
import whisper
import numpy as np
import json
from whisper.tokenizer import LANGUAGES
import torch
import traceback

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
use_fp16 = False
if not (device == 'cpu'):
    # Is gpu
    # If not set to true => torch.OutOfMemoryError: CUDA out of memory.
    use_fp16 = True

#print(LANGUAGES.keys())

def fix_start_end(segments, duration):
    for segment in segments:
        words = segment['words']
        for word in words:
            if word['start'] < 0:
                word['start'] = 0
            if word['end'] > duration:
                word['end'] = duration

def filter_words(segments, threshold):
    def probabilityThreshold(word):
        if word['probability'] > threshold:
            return True
        print(f"Discard {word}")
        return False
    resulting_segments = []
    for segment in segments:
        segment['words'] = list(
            filter(probabilityThreshold, segment['words']))
        if len(segment['words']) > 0:
            resulting_segments.append(segment)
            # fix end and start
            segment['start'] = segment['words'][0]['start']
            segment['end'] = segment['words'][-1]['end']
    return resulting_segments


def join_words(segments):
    transcription2 = ''
    for segment in segments:
        words = segment['words']
        for word in words:
            transcription2 += ' ' + word['word'].strip()
    return transcription2


class Speech2TextProcessor(BaseProcessor):
    async def localConfigure(self):
        # tiny, base, small, medium, large, large-v1, large-v2, large-v3
        self.model_path = "base"
        if ('MODEL' in os.environ):
            self.model_path = os.environ['MODEL']
        print(f"Using model '{self.model_path}'")
        
        # Cargar modelo Whisper
        # model = whisper.load_model("base")
        self.model = whisper.load_model(self.model_path)

    def get_default_arguments(self):
        return {
            'language': 'es',
            'extension': 'mp3',
            'threshold': 0.1,#Define a minimum for word probability
            'gap_ms': 250,# Every time exists words, it gets back this amount
            'threshold_back_ms': 1000,# This define how close a words falls to the end
            'min_duration_ms': 2000,
            'preserve_temp_files': False,
        }

    def print_segments(self, segments):
        for segment in segments:
            self.print_segment(segment)
            print("----------------")

    def print_segment(self, segment):
        words = segment['words']
        for word in words:
            print(f"[{word['start']} - {word['end']}]: {word['word']}")
            
    # https://medium.com/@pouyahallaj/how-to-use-openais-whisper-in-just-3-lines-of-code-for-free-7b5c5dbe4863
    async def process(self, args, default_arguments):
        named_inputs = args['namedInputs']
        buffer = named_inputs['bytes']
        timeline = named_inputs['timeline']
        id = args['id']
        original_t = float(timeline['t'])
        timeline_end = float(timeline['end'])
        audio_duration = timeline_end - original_t
        period = float(timeline['period'])
        
        min_duration_s = default_arguments['min_duration_ms']/1000
        if audio_duration < min_duration_s:
            # should abort
            print(f"Audio duration is {audio_duration}, but at least {min_duration_s} is required")
            return {
                'model_path': self.model_path,
                'timeline': timeline,
                'transcript': {
                    'transcription': "",
                    'segments': []
                }
            }
        
        print(f"[*] processing time from {original_t} to {original_t + period}")

        path = self.store_bytes_locally(
            buffer, f"{default_arguments['extension']}", args)
        #in_memory_file = BytesIO(buffer)
        #in_memory_file = np.frombuffer(buffer, dtype=np.int8)
        # in_memory_file = np.frombuffer(buffer, np.int16).flatten().astype(np.float32)
        # https://github.com/openai/whisper/blob/main/whisper/transcribe.py
        #print(default_arguments)
        try:
            result = self.model.transcribe(
                path,
                #temperature=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5),
                #temperature=(0.0),
                #temperature=(0.0, 0.5),
                temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                word_timestamps=True,
                language=default_arguments['language'],
                condition_on_previous_text=False,
                logprob_threshold=-1.0,
                no_speech_threshold=0.6,  # default is 0.6
                fp16=use_fp16 #FP16 is not supported on CPU; using FP32 instead"
            )
        except Exception as error:
            just_the_string = traceback.format_exc()
            print(just_the_string)
            return {
                'model_path': self.model_path,
                'timeline': timeline,
                'transcript': {
                    'transcription': "",
                    'segments': []
                }
            }
        transcription = result["text"]
        segments = result["segments"]
        #self.print_segments(segments)
        # Erase the sound file
        #if (len(segments) > 0):
        #    self.print_segment(segments[-1])

        # Iterate all segments filtering words threshold
        threshold = default_arguments['threshold']
        gap_ms = default_arguments['gap_ms']
        threshold_back_ms = default_arguments['threshold_back_ms']

        # Fix because some words has end greater than period...
        fix_start_end(segments, period)
        
        time_threshold = period - threshold_back_ms/1000
        #print(f"time_threshold={time_threshold}")
        
        # Filter words below the threshold
        segments = filter_words(segments, threshold)
        
        #print(json.dumps(segments))
        #self.print_segments(segments)
        
        current_size = len(segments)
        #print(f"current_size={current_size}")

        # Recompute new starting point
        if (current_size > 0):
            last_segment = segments[-1]
            # self.print_segment(last_segment)
            #print(json.dumps(last_segment))
            last_segment_end = last_segment['end']
            # Remember
            # last_segment['end'] and last_segment['start'] are time based from 0 to "period"
            # all the times
            if last_segment_end > time_threshold:
                # The next time must start from the end of the last segment
                step_back = max(0, period - last_segment_end)
                timeline['t'] = original_t - step_back
                print(f"[*] next was {original_t+period}, but overwrited to {timeline['t']+period}; stepped back {step_back}")
            else:
                # goes back default time
                step_back = gap_ms/1000
                timeline['t'] = original_t - step_back
                print(f"[*] next was {original_t+period}, but overwrited to {timeline['t']+period}; stepped back {step_back}")
        else:
            print(f"[*] next should be {timeline['t']+period}")

        # Recompute transcription
        transcription = join_words(segments)
        
        print(json.dumps(transcription))
        
        # Last thing, erase file
        if not default_arguments['preserve_temp_files']:
            os.remove(path)

        return {
            'timeline': timeline,
            'transcript': {
                'transcription': transcription,
                'segments': segments
            }
        }


async def main():
    processor = Speech2TextProcessor()
    await processor.configure(sys.argv)
    await processor.start_communication()

if __name__ == '__main__':
    asyncio.run(main())
