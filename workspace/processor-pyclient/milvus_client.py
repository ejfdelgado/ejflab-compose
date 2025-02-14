from milvus_common import MilvusCommon
from pymilvus import MilvusClient, db, FieldSchema, DataType, CollectionSchema, Collection
import os
import logging
from pymilvus.client.types import LoadState
import time
import re
import numpy as np

# https://milvus.io/docs/v2.3.x/boolean.md


class MilvusHandler(MilvusCommon):
    # MilvusHandler.get_client()
    def get_client():
        if not MilvusHandler.client:
            MilvusHandler.initiate_database()
        return MilvusHandler.client
    
    def initiate_database():
        print("Connect to Milvus Database")
        uri = os.getenv('MILVUS_URI', 'http://milvus:19530')
        if "localhost" in uri:
            print("localhost fallback...")
            # weird fallback, it's supposed docker compose take precedence firs .env over host env
            uri = 'http://milvus:19530'
        print(f"MILVUS server on {uri}")
        MilvusHandler.client = MilvusClient(uri)

    def exists_database(name):
        client = MilvusHandler.get_client()
        databases = db.list_database(using=client._using)
        # print(databases)
        return name in databases

    def introspect(self):
        client = MilvusHandler.get_client()
        databases = db.list_database(using=client._using)
        for database in databases:
            print(f"- Database: {database}")
            client.using_database(database)
            collections = client.list_collections()
            for collection in collections:
                print(f"    - Collection: {collection}")

    def use_database(self, name, recreate=False):
        client = MilvusHandler.get_client()
        print(f"Use database {name} recreate? {str(recreate)}...")
        if (not MilvusHandler.exists_database(name)):
            print(f"Creating '{name}' database...")
            db.create_database(db_name=name, using=client._using)
        else:
            if recreate:
                print(f"Recreating '{name}' database...")
                client.using_database(name)
                self.drop_database(name)
                db.create_database(
                    db_name=name, using=client._using)
            else:
                print(f"Using old '{name}' database...")
        client.using_database(name)
        print(f"Use database {name} {str(recreate)}... OK")

    def drop_collection(self, collection_name):
        client = MilvusHandler.get_client()
        print(f"Drop collection {collection_name}...")
        if not client.has_collection(collection_name=collection_name):
            print(f"Drop collection {collection_name} NOT EXISTS...")
            return False
        client.drop_collection(collection_name)
        print(f"Drop collection {collection_name}... OK")
        return True

    def drop_database(self, name):
        client = MilvusHandler.get_client()
        print(f"Drop database {name}...")
        collections = client.list_collections()
        print(
            f"Drop database {name} has {len(collections)} collections inside...")
        for collection_name in collections:
            self.drop_collection(collection_name)
        db.drop_database(db_name=name, using=client._using)
        print(f"Drop database {name} OK!")

    async def wait_release(self, collection_name):
        client = MilvusHandler.get_client()
        print(f"Wait release {collection_name}...")
        client.release_collection(
            collection_name=collection_name
        )
        while (True):
            res = client.get_load_state(
                collection_name=collection_name
            )
            print(f"Wait release {collection_name}... {res['state']}")
            if res['state'] == LoadState.NotLoad:
                break
            time.sleep(1)
        print(f"Wait released {collection_name}... OK")
        return True

    async def wait_loaded(self, collection_name):
        client = MilvusHandler.get_client()
        print(f"Wait loaded {collection_name}...")
        if not client.has_collection(collection_name=collection_name):
            return False

        client.load_collection(collection_name)
        while (True):
            res = client.get_load_state(
                collection_name=collection_name
            )
            print(f"Wait loaded {collection_name}... {res['state']}")
            if res['state'] == LoadState.Loaded:
                break
            time.sleep(1)
        print(f"Wait loaded {collection_name}... OK")
        return True

    async def describe(self, collection_name):
        client = MilvusHandler.get_client()
        await self.wait_loaded(collection_name)
        res = client.describe_collection(
            collection_name=collection_name
        )
        print(res)

    async def create_collection_with_schema(self, collection_name):
        client = MilvusHandler.get_client()
        print(f"Creating collection {collection_name}...")
        id_field = FieldSchema(
            name="id",
            dtype=DataType.INT64,
            is_primary=True,
            description="primary id"
        )
        speaker_field = FieldSchema(
            name="speaker",
            dtype=DataType.VARCHAR,
            description="speaker",
            max_length=32
        )
        embedding_field = FieldSchema(
            name="embedding",
            dtype=DataType.FLOAT_VECTOR,
            dim=512,
            description="vector"
        )

        schema = CollectionSchema(
            fields=[id_field, speaker_field, embedding_field],
            auto_id=True,
            enable_dynamic_field=True,
            description="helps diarization",
            # https://milvus.io/docs/use-partition-key.md#Use-Partition-Key
            partition_key_field="speaker",
        )

        index_params = client.prepare_index_params()

        index_params.add_index(
            field_name="id",
            index_type="STL_SORT"
        )

        index_params.add_index(
            name='vector_idx',
            field_name="embedding",
            index_type="IVF_FLAT",
            metric_type="COSINE",  # COSINE, L2, IP, JACCARD, HAMMING
            params={"nlist": 16}  # Number of cluster units
        )

        client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params,
            shards_num=3,
            num_partitions=16  # Number of partitions. Defaults to 16 or 64?.
        )

        print(f"Creating collection {collection_name}... OK!")

    def insert(self, data, collection_name, db_name=''):
        client = MilvusHandler.get_client()
        res = client.insert(
            collection_name=collection_name,
            data=data
        )
        # print(res)

    def search(
            self, 
            embeed, 
            collection_name, 
            db_name='', 
            search_params = {
                # "radius": 0.8, # Radius of the search circle
                "metric_type": "COSINE",  # COSINE, L2, IP, JACCARD, HAMMING
                "params": {

                }
            },
            output_fields = ['id', 'speaker'],
            consistency_level="Strong",
            limit=1,
            offset=0# Where to place this??
        ):
        client = MilvusHandler.get_client()
        
        res = client.search(
            collection_name=collection_name,
            data=[embeed],
            # https://milvus.io/api-reference/pymilvus/v2.3.x/ORM/Collection/search.md
            # https://milvus.io/docs/v2.3.x/partition_key.md
            # expr=f"speaker=='some_value'",
            limit=limit,
            consistency_level=consistency_level,
            search_params=search_params,
            output_fields=output_fields
        )
        return res
    
    def search_faces(self, embeed, collection_name, db_data, paging, extra):
        client = MilvusHandler.get_client()
        search_params = {
            "metric_type": "IP",  # COSINE, L2, IP, JACCARD, HAMMING
            "params": {
                # While radius sets the outer limit of the search, 
                # range_filter can be optionally used to define an inner boundary, 
                # creating a distance range within which vectors must fall to be considered matches.
                #"range_filter": 1.0
            }
        }
        if 'minDistance' in extra and isinstance(extra['minDistance'], (int, float)):
            # Defines the outer boundary of your search space. 
            # Only vectors that are within this distance from 
            # the query vector are considered potential matches.
            search_params['params']['radius'] = extra['minDistance']
            print(f"Using radius of {extra['minDistance']}")
        res = client.search(
            collection_name=collection_name,
            data=[embeed],
            # https://milvus.io/docs/v2.3.x/partition_key.md
            # expr=f"speaker=='some_value'",
            limit=paging['limit'],
            offset=paging['offset'],
            consistency_level="Strong",
            search_params=search_params,
            output_fields=['id', 'document_id', 'face_path', 'millis', 'x1', 'y1', 'x2', 'y2', 'ref_id']
        )
        response = []
        for hits in res:
            for hit in hits:
                response.append(hit)

        return response

    def count(self, collection_name):
        client = MilvusHandler.get_client()
        res = client.query(
            collection_name=collection_name,
            output_fields=["count(*)"],
            consistency_level="Strong"
        )
        print(res[0])
        return res[0]['count(*)']

try:
    MilvusHandler.initiate_database()
except Exception as error:
    print(error)