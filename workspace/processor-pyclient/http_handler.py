from functools import cached_property
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qsl, urlparse
from http.cookies import SimpleCookie
import json
import threading
from time import sleep
import asyncio
import msgpack
import traceback
import types
import os

class HttpHandler(BaseHTTPRequestHandler):
    processor = None

    @cached_property
    def url(self):
        return urlparse(self.path)

    @cached_property
    def query_data(self):
        return dict(parse_qsl(self.url.query))

    @cached_property
    def post_data(self):
        content_length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(content_length)

    @cached_property
    def form_data(self):
        # return dict(parse_qsl(self.post_data.decode("utf-8")))
        return None

    @cached_property
    def cookies(self):
        return SimpleCookie(self.headers.get("Cookie"))

    def custom_json_parse(self, text):
        try:
            return json.loads(text)
        except:
            return text

    def do_POST(self):
        self.do_GET()

    def stream_response(self, generator):
        try:
            print("Stream response...")
            self.send_response(200)
            self.send_header("Content-type", "text/plain; charset=utf-8")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            
            for pal in generator:
                texto = pal.encode("utf-8")
                tamanio = f"{len(texto):X}\r\n".encode("utf-8")
                self.wfile.write(tamanio)  # Chunk size in hex
                self.wfile.write(texto)
                self.wfile.write(b"\r\n")  # Chunk terminator
                self.wfile.flush()
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except BrokenPipeError:
            print("Client disconnected.")
        except Exception as e:
            traceback.print_exc()
    
    def do_GET(self):
        correcto = True
        try:
            respuesta = self.get_response()
            if isinstance(respuesta, types.GeneratorType):
                self.stream_response(respuesta)
                return
            message = json.dumps(respuesta).encode("utf-8")
        except Exception as e:
            correcto = False
            message = "Error {0}".format(str(e.args[0])).encode("utf-8")
        if correcto:
            self.send_response(200)
        else:
            self.send_response(400)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(message)

    def between_callback(original_arguments):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        some_callback = original_arguments['thread_function']
        args = original_arguments['my_inputs']
        future_response = loop.run_until_complete(some_callback(args))
        loop.close()
        return future_response

    def fire_thread(thread_function, my_inputs):
        argumentos = {
            'thread_function': thread_function,
            'my_inputs': my_inputs
        }
        x = threading.Thread(target=HttpHandler.between_callback,
                             args=(argumentos,))
        x.start()
        return {'started': True}

    async def local_exec(self, input):
        loaded = input['post_data']
        loaded['path'] = input['path']
        response = await self.processor.process_local(loaded)
        return response

    def get_response(self):
        try:
            data_type = "msgpack"
            if "data_type" in self.query_data:
                data_type = self.query_data['data_type']
            if data_type == "msgpack":
                post_data = msgpack.unpackb(self.post_data)
            elif data_type == "json":
                post_data = json.loads(self.post_data)
            input = {
                "path": self.url.path,
                "query_data": self.query_data,
                "post_data": post_data,
                "form_data": self.form_data,
                "cookies": {
                    name: cookie.value
                    for name, cookie in self.cookies.items()
                },
            }
            if (self.url.path.endswith("/syncprocess")):
                argumentos = {
                    'thread_function': self.local_exec,
                    'my_inputs': input
                }
                return HttpHandler.between_callback(argumentos)
            if (self.url.path.endswith("/process")):
                return HttpHandler.fire_thread(self.local_exec, input)
            else:
                return input
        except Exception as error:
            traceback.print_exc()
            return {}
