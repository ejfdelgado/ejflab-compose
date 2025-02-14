from abc import ABC, abstractmethod
import os
import subprocess
from http.server import HTTPServer
from http_handler import HttpHandler
import logging
import requests
import socketio
import msgpack
import io
import base64
import sys
import re
import asyncio
import threading
from datetime import datetime
import traceback
import urllib3

urllib3.disable_warnings()

# Interaction with pub/sub
# https://cloud.google.com/run/docs/tutorials/pubsub#run_pubsub_handler-python

mutex = threading.Lock()

class BaseProcessor(ABC):

    def __init__(self):
        format = "%(asctime)s: %(message)s"
        logging.basicConfig(format=format, level=logging.INFO,
                            datefmt="%H:%M:%S")
        self.channel = os.environ['CHANNEL']
        self.channel_servers = os.environ['CHANNEL_SERVERS']

    async def start_communication(self):
        if "websocket" in self.channel_servers:
            HttpHandler.fire_thread(self.start_web_soket, {})
        if "post" in self.channel_servers:
            HttpHandler.fire_thread(self.start_http_server, {})

    async def process_local(self, args):
        print(f"process_local {args['room']}")
        id = args['id']
        room = args['room']
        processorId = args['processorMethod']
        default_arguments = self.get_default_arguments()
        if "data" in args:
            temp = args['data']
            if temp:
                for llave in temp:
                    default_arguments[llave] = temp[llave]
        if "general" in args:
            temp = args['general']
            if temp:
                for llave in temp:
                    default_arguments[llave] = temp[llave]
        print(f"{processorId} start processing...")    
        try:
            mutex.acquire()
            if processorId == "health":
                respuesta = await self.process_health(args, default_arguments)
            elif processorId == "python":
                respuesta = await self.process_python(args, default_arguments)
            elif processorId == "upload_file":
                respuesta = await self.process_upload_file(args, default_arguments)
            else:
                respuesta = await self.process(args, default_arguments)
            print(f"{processorId} processing... OK!")
        except Exception as error:
            just_the_string = traceback.format_exc()
            message = f"{processorId} error: {just_the_string}"
            print(message)
            respuesta = {"error": message}
        finally:
            mutex.release()
        print(f"{processorId} sending response...")
        pub_res = await self.publish_response(id, processorId, respuesta, room, args)
        print(f"{processorId} sending response... OK!")
        return pub_res

    async def process_health(self, args, default_arguments):
        named_inputs = args['namedInputs']
        command = named_inputs['command']
        print(f"command {command}")
        pipe = subprocess.run(["bash", "-c", command],
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE,
                              universal_newlines=True,
                              bufsize=10**8)
        stdout = pipe.stdout
        stderr = pipe.stderr

        return {
            'command': command,
            'stdout': stdout,
            'stderr': stderr,
            }
        
    async def process_upload_file(self, args, default_arguments):
        named_inputs = args['namedInputs']
        bytes = named_inputs['bytes']
        file_name = named_inputs['file_name']
        
        file = open(f"temp/{file_name}", "wb")
        file.write(bytes)
        file.close()
        
        return {}
        
    async def process_python(self, args, default_arguments):
        named_inputs = args['namedInputs']
        code = named_inputs['command']
        file = open("./temp/eraseme.py", "w")
        file.write(code)
        file.close()
        command = "python3 ./temp/eraseme.py"
        pipe = subprocess.run(["bash", "-c", command],
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE,
                              universal_newlines=True,
                              bufsize=10**8)
        stdout = pipe.stdout
        stderr = pipe.stderr

        return {
            'command': command,
            'stdout': stdout,
            'stderr': stderr,
            }
        
    async def start_web_soket(self, input):
        self.sio = socketio.AsyncClient(
            reconnection=True,
            reconnection_attempts=0,  # Infinite
            # Each successive attempt doubles this delay.
            reconnection_delay=1,
            reconnection_delay_max=5,
            randomization_factor=0.5,
            logger=False
        )

        @self.sio.event
        async def connect():
            await self.connect()

        @self.sio.event
        async def disconnect():
            self.disconnect()

        @self.sio.event
        async def flowchartLoaded(args):
            await self.register_me()

        @self.sio.event
        async def process(args):
            await self.process_local(args)

        await self.sio.connect(self.serverws, headers={'room': self.room})
        await self.sio.wait()

    def start_http_server(self, input):
        HttpHandler.processor = self
        self.server = HTTPServer(
            ("0.0.0.0", int(os.environ['PORT'])), HttpHandler)
        logging.info(
            f"Http server on {os.environ['PORT']}... :) in {os.getcwd()}")
        self.server.serve_forever()

    @classmethod
    @abstractmethod
    async def localConfigure(self):
        pass

    @classmethod
    @abstractmethod
    async def process(self, args):
        pass

    @classmethod
    @abstractmethod
    def get_default_arguments(self):
        return {}

    async def publish_response(self, id, processorId, respuesta, topic, input_args):
        inputs = []
        if 'inputs' in input_args:
            inputs = input_args['inputs']
        payload = {
            'processorId': processorId,
            'id': id,
            'data': respuesta,
            'room': topic,
            'inputs': inputs
        }
        if "path" in input_args:
            if input_args["path"].endswith("/syncprocess"):
                return respuesta

        if self.channel == "websocket":
            await self.sio.emit("processResponse", payload)
        elif self.channel == "queue":
            topic_url = os.environ['SERVER_PUB_SUB_URL']
            # Publish into the queue the event
            # The server must know if discharge the message if it has not the room
            pass
        elif self.channel == "post":
            server_url = os.environ['SERVER_POST_URL']
            # Make the post to the server
            url = f"{server_url}/srv/flowchart/processor_response"
            response = BaseProcessor.make_post_to_url(url, payload)
            status = response.status_code
            if (status != 200):
                statusText = response.reason
                print(f"Server Error: {statusText}")

    def make_post_to_url(url, payload):
        my_bytes = msgpack.packb(payload)
        print(f"Sending {str(len(my_bytes))} bytes")
        response = requests.post(url, data=my_bytes, headers={
                                    'Content-Type': 'application/octet-stream'}, verify=False)
        return response        
    
    def save_bytes(self, path, buffer):
        with open(path, "wb") as binary_file:
            binary_file.write(buffer)

    async def connect(self):
        logging.info('websocket connected to server')
        await self.register_me()

    def disconnect(self):
        logging.info('websocket disconnected from server')

    async def configure(self, args):
        self.uid = os.environ['PROCESSOR_UID']
        self.serverws = os.environ['SERVER_WS']
        self.room = os.environ['ROOM']
        logging.info(
            f"uid {self.uid} room {self.room} serverws {self.serverws}")
        await self.localConfigure()

    async def register_me(self):
        await self.sio.emit("registerProcessor", {
            'uid': self.uid,
        })

    async def readVideo(self, timeline, source, inputs):
        print(f"Read Video!")
        t = timeline['t']
        to = t + timeline['period']
        fps = float(source['fps'])
        period = 1/fps
        local_t = t
        response = []
        print(f"from {t} to {to} every {period}")
        while True:
            if local_t >= to:
                break
            timeline['t'] = local_t
            bytes = await self.readImage(timeline, source)
            response.append(bytes)
            local_t = local_t + period
        return response

    async def readImage(self, timeline, source, inputs):
        print(f"Read Image")
        t = timeline['t']
        offset = 0
        if len(inputs) > 0 and 'dperiod' in timeline:
            offset = inputs[0] * timeline['dperiod']
        print(f"Reading image from {t} to {t - offset}")
        t = t - offset
        
        src = source['src']
        width = "-1"
        height = "-1"
        if "width" in source:
            width = source['width']
        if "height" in source:
            height = source['height']
        ffmpeg_command = ["ffmpeg",
                          "-ss", f"{t}",  # Seek input
                          "-i", src,  # The source file
                          "-an",  # No Audio
                          "-y",  # overwrite output files
                          "-f", "image2pipe",
                          "-c:v", "png",
                          "-vframes", "1"
                          ]
        if not width == "-1" or not height == "-1":
            ffmpeg_command.append("-vf")
            ffmpeg_command.append(f"scale={width}:{height}")

        ffmpeg_command.append("pipe:1")

        pipe = subprocess.run(ffmpeg_command,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE,
                              bufsize=10**8)
        reference = io.BytesIO(pipe.stdout)
        bytes = reference.read()
        if source['base64']:
            encoded = base64.b64encode(bytes)
            print(
                f"data:image/png;base64,{encoded.decode('utf-8')}")
        return bytes

    async def get_video_attributes(self, source, media):
        src = source['src']
        if not os.path.isfile(src):
            raise ValueError(f"File {src} not exists!")
        exiftool_command = [
            "exiftool", src
        ]
        exiftool_pipe = subprocess.run(exiftool_command,
                                       stdout=subprocess.PIPE,
                                       universal_newlines=True
                                       )
        complet_text = exiftool_pipe.stdout

        print(complet_text)

        def extract_part(exp, filter=None):
            format_str = "%Y:%m:%d %H:%M:%S%z"
            format_str_2 = "%Y:%m:%d %H:%M:%S"
            p = re.compile(f"{exp}\s*:\s*(.+)", re.IGNORECASE | re.MULTILINE)
            match = p.search(complet_text)
            try:
                if match:
                    part = match.group(1)
                    if filter == "datetime":
                        if part == "0000:00:00 00:00:00":
                            return 0
                        try:
                            part = datetime.strptime(part, format_str)
                        except ValueError:
                            part = datetime.strptime(part, format_str_2)
                        #Using millis
                        return int(part.timestamp())*1000
                    elif filter == 'int':
                        return int(part)
                    elif filter == 'float':
                        return float(part)
                    elif filter == 'time':
                        print(f"{part}")
                        # Clean
                        part = part.replace(' (approx)', '')
                        p2 = re.compile(f"^(\d+(\.\d+)?)\s*s$",
                                        re.IGNORECASE | re.MULTILINE)
                        match2 = p2.search(part)
                        if (match2):
                            s = match2.group(1)
                            return float(s)
                        # Try whole
                        h, m, s = map(float, part.split(':'))
                        print(f"{h} {m} {s}")
                        part = h * 3600 + m * 60 + s
                        return part
                    return part
                else:
                    return None
            except Exception as error:
                just_the_string = traceback.format_exc()
                print(just_the_string)
                return None

        complete = {
            # Useless data because are date from file
            "date_modified_content": extract_part('File Modification Date/Time', 'datetime'),
            "date_modified_inode": extract_part('File Inode Change Date/Time', 'datetime'),
            # Usefull date because is the camera date
            "date_created": extract_part('Create Date', 'datetime'),
            "date_modified": extract_part('Modify Date', 'datetime'),
            # Usefull date because is the camera date
            "date_track_created": extract_part('Track Create Date', 'datetime'),
            "date_track_modified": extract_part('Track Modify Date', 'datetime'),
            # Usefull date because is the camera date
            "date_media_created": extract_part('Media Create Date', 'datetime'),
            "date_media_modified": extract_part('Media Modify Date', 'datetime'),
            "file_name": extract_part('File Name'),
            "mime_type": extract_part('MIME Type'),
            "size_bytes": extract_part('Media Data Size', 'int'),
            "seconds": extract_part('Duration', 'time'),
            "encoder": extract_part('Encoder'),
            "image": {
                "width": extract_part('Image Width', 'int'),
                "height": extract_part('Image Height', 'int'),
                "x_resolution": extract_part('X Resolution', 'int'),
                "y_resolution": extract_part('Y Resolution', 'int'),
                "bit_dept": extract_part('Bit Depth', 'int'),
                "frame_rate_hz": extract_part('Video Frame Rate', 'float'),
                "pixel_ratio": extract_part('Pixel Aspect Ratio'),
                # https://sirv.com/help/articles/rotate-photos-to-be-upright/
                "rotation": extract_part('Rotation', 'int'),
            },
            "audio": {
                "format": extract_part('Audio Format'),
                "channels": extract_part('Audio Channels', 'int'),
                "bits_per_sample": extract_part('Audio Bits Per Sample', 'int'),
                "sample_rate_hz": extract_part('Audio Sample Rate', 'int'),
            }
        }
        if 'startTime' in media and isinstance(media['startTime'], (int, float)):
            print("Got start date from input ignoring what the video metadata sais")
            complete['date_created'] = media['startTime']
        if complete['date_created'] == 0:
            complete['date_created'] = complete['date_track_created']
        if complete['date_created'] == 0:
            complete['date_created'] = complete['date_media_created']
        if not isinstance(complete['date_created'], (int, float)) or complete['date_created'] <= 0:
            raise Exception("Video start date could not be infered, but needed")
        
        # Add thumbnail location
        complete['thumbnail'] = {
            "t": complete["seconds"]/2
        }
        
        #print(complete)

        return complete

    async def mp3_to_wav(self, mp3Path):
        wavPath = re.sub(r"\.mp3$", ".wav", mp3Path, flags=re.IGNORECASE)
        ffmpeg_command = ["ffmpeg",
                          "-i", mp3Path,
                          "-acodec", "pcm_u8",
                          # audio sampling rate (in Hz)
                          "-ar", "16000",
                          # number of audio channels
                          "-ac", "1",
                          wavPath
                          ]

        pipe = subprocess.run(ffmpeg_command,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE,
                              bufsize=10**8)
        return wavPath

    async def readAudio(self, timeline, source, inputs):
        t = timeline['t']
        duration_seconds = timeline['period']
        src = source['src']
        bits_per_second = source['bitsPerSecond']
        format = source["format"]
        ffmpeg_command = ["ffmpeg",
                          "-ss", f"{t}",  # Seek input
                          "-i", src,  # The source file
                          "-ab", bits_per_second,
                          # "-acodec", "pcm_s16le",
                          # "-ar", "1", # audio sampling rate (in Hz)
                          # "-ac", "0",  # number of audio channels
                          # "0:a",
                          # "-map_metadata", "-1",
                          # "-sn",  # disable subtitle
                          "-vn",  # No Video
                          "-y",  # overwrite output files
                          "-f", format,
                          "-t", f"{duration_seconds}",
                          "pipe:1"]
        pipe = subprocess.run(ffmpeg_command,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE,
                              bufsize=10**8)
        reference = io.BytesIO(pipe.stdout)
        bytes = reference.read()
        if source['base64']:
            encoded = base64.b64encode(bytes)
            print(
                f"data:audio/{source['format']};base64,{encoded.decode('utf-8')}")
        return bytes

    async def read(self, timeline, source, data, inputs):
        type = source['type']
        if type == "audio":
            return await self.readAudio(timeline, source, inputs)
        elif type == "image":
            return await self.readImage(timeline, source, inputs)
        elif type == "video":
            return await self.readVideo(timeline, source, inputs)

    def store_bytes_locally(self, buffer, extension, args, suffix=""):
        file_name = f"{args['id']}{suffix}.{extension}"
        root_dir = f"./temp/{args['room']}/{os.environ['PROCESSOR_UID']}"
        os.makedirs(root_dir, exist_ok=True)
        path = f"{root_dir}/{file_name}"
        print(f"Writing {str(len(buffer))} Bytes into {path}")
        self.save_bytes(path, buffer)
        return path

    def store_mpk_locally(self, my_json, file_name, args):
        my_bytes = msgpack.packb(my_json)
        root_dir = f"./temp/{args['room']}/{os.environ['PROCESSOR_UID']}"
        os.makedirs(root_dir, exist_ok=True)
        path = f"{root_dir}/{file_name}"
        self.save_bytes(path, my_bytes)
        return path

    def numpy_array_to_native_array(self, some_array):
        response = []
        #print(type(some_array))
        #print(some_array.shape)
        for elem in some_array:
            response.append(elem.item())
        return response

#{
#    "mediaStartTime":1729374530919,
#    "mediaEndTime":1729374530919,
#    "mediaSourceUrl":"/local/video.mp4",
#    "imageSourceUrl":"/local/big.jpg",
#    "thumbnail":"/local/small.jpg",
#    "frameWidth":100,
#    "frameHeight":100,
#    "imageBboxX1":0.3,
#    "imageBboxX2":0.5,
#    "imageBboxY1":0.1,
#    "imageBboxY2":0.15,
#    "imageBboxScore":0.1,
#    "text":"Some text",
#    "textScore":0.2
#}
async def register_face_found(db_data, extra):
    return await register_object_found(db_data, 'FACE', extra)
async def register_vehicle_found(db_data, extra):
    return await register_object_found(db_data, 'VEHICLE', extra)
async def register_plate_found(db_data, extra):
    return await register_object_found(db_data, 'LICENCE_PLATE', extra)
async def register_intent_found(db_data, extra):
    return await register_object_found(db_data, 'INTENT', extra)

async def register_object_found(db_data, objectTypeId, extra):
    server_url = os.environ['SERVER_POST_URL']
    # Make the post to the server
    url = f"{server_url}/srv/widesight/accounts/{db_data['accountId']}/apps/{db_data['appId']}/objects/add"
    extra['mediaId'] = db_data['mediaId']
    extra['mediaType'] = db_data['mediaType'].upper()
    extra['objectTypeId'] = objectTypeId
    response = requests.post(url, json=extra, headers={
                                'Content-Type': 'application/json'}, verify=False)
    status = response.status_code
    # response.object.id
    # response.inserted
    if (status != 200):
        statusText = response.reason
        print(f"Server Error: {statusText}")
        return None
    return response.json()
