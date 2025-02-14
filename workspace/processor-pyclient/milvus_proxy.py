from base_procesor import BaseProcessor
from milvus_common import MilvusCommon
import os
import json
import numpy as np

class MilvusProxy(MilvusCommon):
    def insert(self, data, collection_name, db_name = "searchable"):
        print("insert")
        payload = {
            'db_name': db_name,
            'collection_name': collection_name,
            'data': data,
        }
        server_url = os.environ['SERVER_POST_URL']
        # Make the post to the server
        url = f"{server_url}/srv/milvus/insert"
        response = BaseProcessor.make_post_to_url(url, payload)
        content = response.content
        print(content)
        return None
    
    def search(
            self, 
            embeed, 
            collection_name, 
            db_name = "searchable",
            search_params = {
                # "radius": 0.8, # Radius of the search circle
                "metric_type": "COSINE",  # COSINE, L2, IP, JACCARD, HAMMING
                "topk": 1,
                "params": '{"nprobe": 1024}'
            },
            output_fields = ['id', 'speaker'],
            consistency_level="Strong",
            limit=1,
            offset=0
        ):
        payload = {
            'db_name': db_name,
            'collection_name': collection_name,
            'embeed': self.numpy_array_to_native_array(embeed),
            'paging': {
                'limit': limit,
                'offset': offset,
            },
            'search_params': search_params,
            'output_fields': output_fields,
            'consistency_level': consistency_level,
        }
        server_url = os.environ['SERVER_POST_URL']
        # Make the post to the server
        url = f"{server_url}/srv/milvus/search"
        response = BaseProcessor.make_post_to_url(url, payload)
        #content = response.content
        #print("Content:")
        content = json.loads(response.text)
        print("content")
        print(content)
        return content['results']
    
    def search_faces(self, embeed, collection_name, db_data, paging, extra):
        print("search_faces")
        db_name = "searchable"
        payload = {
            'db_name': db_name,
            'embeed': embeed,
            'collection_name': collection_name,
            'db_data': db_data,
            'paging': paging,
            'extra': extra
        }
        server_url = os.environ['SERVER_POST_URL']
        # Make the post to the server
        url = f"{server_url}/srv/milvus/search"
        response = BaseProcessor.make_post_to_url(url, payload)
        content = json.loads(response.text)
        #print(content['results'])
        return content['results']
    
    def use_database(self, name, recreate=False):
        pass
    
    async def wait_loaded(self, collection_name):
        pass
    
