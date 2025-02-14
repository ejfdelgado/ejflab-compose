import asyncio
import sys
import os
from milvus_client import MilvusHandler
from milvus_proxy import MilvusProxy
import json
import numpy as np
from pyannote.audio import Pipeline, Inference, Model, Audio
from pyannote.core import Segment
from scipy.spatial.distance import cdist


token = os.environ['PYANNOTE_TOKEN']
# Accepted TOS
# https://huggingface.co/pyannote/speaker-diarization-3.1
# https://hf.co/pyannote/segmentation-3.0
print(f"Pipeline.from_pretrained...")
pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1", use_auth_token=token)

print(f"Model.from_pretrained...")
model = Model.from_pretrained("pyannote/embedding",
                              use_auth_token=token)

np.set_printoptions(precision=3, suppress=True)

print(f"Ready!")

def get_words_from_to(segments, start_time, end_time):
    my_words = []
    for segment in segments:
        words = segment['words']
        for word in words:
            word_start = word['start']
            word_end = word['end']
            #if start_time <= word_start and word_end <= end_time:
            if not ('used' in word) and start_time < word_end and word_start < end_time:
                my_words.append(word['word'].strip())
                word['used'] = True
    phrase = " ".join(my_words)
    #print(f"[{start_time} - {end_time}]: {phrase}")
    return phrase

SUPER_CLASS = MilvusHandler
if 'MILVUS_PROXY' in os.environ and os.environ['MILVUS_PROXY'] == '1':
    SUPER_CLASS = MilvusProxy

print(f"Extending from {SUPER_CLASS}")

class DiarizationProcessor(SUPER_CLASS):
    async def localConfigure(self):
        pass

    def get_default_arguments(self):
        return {
            'speaker_prefix': 'Persona',
            'min_embed_ms': 1500,
            'speaker_threshold': 0.3,  # 1 means same voice, -1 means different voice
            'preserve_temp_files': False,
            'merge_same_speaker': True,
            'index_all_words': False,
            'use_speaker_cache': True,
        }

    async def diarization(self, args, default_arguments):
        data = args['data']
        named_inputs = args['namedInputs']
        transcript = named_inputs['transcript']
        timeline = named_inputs['timeline']
        my_bytes = named_inputs['bytes']
        period_time = timeline['period']
        offset_time = timeline['t']
        end_time_global = timeline['end']
        MIN_SIZE = default_arguments['min_embed_ms']/1000
        speaker_threshold = default_arguments['speaker_threshold']
        merge_same_speaker = default_arguments['merge_same_speaker']
        index_all_words = default_arguments['index_all_words']
        use_speaker_cache = default_arguments['use_speaker_cache']
        diarization_data = named_inputs['diarization']
        segments = transcript['segments']
        room = args['room']
        cache_speakers = []

        db_name = data["db"]
        collection_name = data["collection"]
        complete_mp3_file_path = None
        complete_wav_file_path = None

        gap_analysed_time = end_time_global - offset_time
        if gap_analysed_time < MIN_SIZE:
            print(
                f"Can't process {gap_analysed_time} seconds because it needs at least {MIN_SIZE} seconds")
            diarization_data['partial'] = []
            return {
                'diarization': diarization_data
            }
        if len(segments) == 0:
            diarization_data['partial'] = []
            return {
                'diarization': diarization_data
            }

        try:
            self.use_database(db_name, False)

            # print(f"nspeakers = {diarization_data['nspeakers']}")

            # Stores locally the my_bytes
            complete_mp3_file_path = self.store_bytes_locally(
                my_bytes, "mp3", args)
            # Convert mp3 to wav
            complete_wav_file_path = await self.mp3_to_wav(complete_mp3_file_path)

            # Get duration in seconds
            audio_durations_seconds = Audio().get_duration(
                complete_wav_file_path)
            #print(f"audio_durations_seconds: {audio_durations_seconds}")
            print("diarization start!")
            diarization = pipeline(complete_wav_file_path)
            print("diarization end!")
            diarization_results = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                start_time, end_time = turn.start, turn.end
                end_time = min(period_time, end_time, audio_durations_seconds)
                max_limit = min(period_time, audio_durations_seconds)
                # Fix start or end time if smaller than 1.5 seconds
                time_span = end_time - start_time
                needed_half_aditional = (MIN_SIZE - time_span)/2
                if needed_half_aditional > 0:
                    # print(f"FIX {start_time} - {end_time}")
                    start_time = start_time - needed_half_aditional
                    end_time = end_time + needed_half_aditional
                    if start_time < 0:
                        end_time = end_time - start_time
                        start_time = 0
                    elif end_time > max_limit:
                        diff = end_time - max_limit
                        start_time = start_time - diff
                        end_time = max_limit
                    # print(f"FIXED {start_time} - {end_time}")

                segment_text = get_words_from_to(segments, start_time, end_time)
                if len(segment_text.strip()) > 0:
                    # print(f'turn: {turn} {speaker} {start_time} -> {end_time}')
                    inference = Inference(model, window="whole")
                    excerpt = Segment(start_time, end_time)
                    embedding = inference.crop(complete_wav_file_path, excerpt)

                    speaker_name = f"{default_arguments['speaker_prefix']}{diarization_data['nspeakers']+1}"
                    distance = -1
                    speaker_found = ""
                    
                    is_new_voice = True
                    
                    # Search in cache
                    if use_speaker_cache:
                        for old in cache_speakers:
                            # <class 'numpy.float32'>
                            distance = self.cosine_distance(old['embedding'], embedding)
                            distance = distance.item()
                            #distance = np.float16(distance)
                            if distance >= speaker_threshold:
                                speaker_found = old['speaker']
                                print(f"Cache hit for {speaker_found} with distance {distance}")
                                is_new_voice = False
                                break

                    if speaker_found == "":
                        closest = self.search(embedding, collection_name, db_name)
                            
                        for my_close in closest:
                            if (len(my_close) == 0):
                                break
                            my_close = my_close[0]
                            entity = my_close['entity']
                            speaker_found = entity['speaker']
                            distance = my_close['distance']
                            if distance < speaker_threshold:
                                # No similar voice exists before
                                diarization_data['nspeakers'] = diarization_data['nspeakers'] + 1
                                speaker_name = f"{default_arguments['speaker_prefix']}{diarization_data['nspeakers']+1}"
                            else:
                                # Exist similar voice
                                print(f"Milvus hit for {speaker_found} with distance {distance}")
                                speaker_name = speaker_found
                                is_new_voice = False

                    embedding_native = embedding
                    if issubclass(self.__class__, MilvusProxy):
                        # MilvusProxy needs change in the arrays
                        embedding_native = self.numpy_array_to_native_array(embedding)
                    
                    if use_speaker_cache:
                        cache_speakers.append({
                            'speaker': speaker_name,
                            'embedding': embedding
                        })
                    
                    if is_new_voice or index_all_words:
                        self.insert({
                            'speaker': speaker_name,
                            'embedding': embedding_native
                        }, collection_name, db_name)
                    
                    added_new = {
                        "document_id": room,
                        "start_time": start_time + offset_time,
                        "end_time": end_time + offset_time,
                        "speaker": speaker_name,
                        "text": segment_text,
                        "distance": distance,
                        "distance_from": speaker_found,
                    }
                    
                    is_appended = False
                    if merge_same_speaker:
                        if len(diarization_results) > 0:
                            previous = diarization_results[-1]
                            if previous['speaker'] == added_new['speaker']:
                                previous['end_time'] = added_new['end_time']
                                previous['text'] = previous['text'] + ' ' + added_new['text']
                                is_appended = True
                            
                    if not is_appended:
                        diarization_results.append(added_new)

            print("Finishing diarization...")
            if not default_arguments['preserve_temp_files']:
                print("Deleteing files")
                os.remove(complete_mp3_file_path)
                os.remove(complete_wav_file_path)
            diarization_data['partial'] = diarization_results
            return {
                'diarization': diarization_data
            }
        except Exception as error:
            print(f"Bytes len {len(my_bytes)}")
            print(f"complete_mp3_file_path = {complete_mp3_file_path}")
            print(f"complete_wav_file_path = {complete_wav_file_path}")
            print(timeline)
            raise error

    async def process(self, args, default_arguments):
        method = args['method']
        if method == "diarization":
            return await self.diarization(args, default_arguments)
        return {}


async def main():
    processor = DiarizationProcessor()
    await processor.configure(sys.argv)
    await processor.start_communication()

if __name__ == '__main__':
    asyncio.run(main())
