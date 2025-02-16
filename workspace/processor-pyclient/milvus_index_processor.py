import asyncio
import sys
import json
import os
import re
import asyncio
from milvus_client import MilvusHandler
from milvus_proxy import MilvusProxy
from FlagEmbedding import FlagModel
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from pymilvus.model.reranker import BGERerankFunction
from pymilvus import (
    model, AnnSearchRequest, RRFRanker, Collection, utility, connections
)
from base_procesor import register_intent_found
import gc
import torch
import numpy as np
import uuid

import asyncpg

#gc.collect()
#torch.cuda.empty_cache()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
use_fp16 = False
if not (device == 'cpu'):
    # Is gpu
    # If not set to true => torch.OutOfMemoryError: CUDA out of memory.
    use_fp16 = True

print(f"device {device} use_fp16 {use_fp16}")
# https://github.com/FlagOpen/FlagEmbedding
# https://huggingface.co/BAAI/bge-m3
# https://github.com/milvus-io/pymilvus/blob/master/examples/hello_hybrid_sparse_dense.py
# https://github.com/FlagOpen/FlagEmbedding/tree/master/FlagEmbedding/BGE_M3#usage
bge_m3_ef = BGEM3EmbeddingFunction(
    model_name='BAAI/bge-m3',  # Specify the model name
    device=f"{device}",  # Specify the device to use, e.g., 'cpu' or 'cuda:0'
    # Specify whether to use fp16. Set to `False` if `device` is `cpu`.
    use_fp16=use_fp16
)
dense_dim = bge_m3_ef.dim["dense"]

print(f"dense_dim = {dense_dim}")

ef = model.DefaultEmbeddingFunction()

SUPER_CLASS = MilvusHandler
if 'MILVUS_PROXY' in os.environ and os.environ['MILVUS_PROXY'] == '1':
    SUPER_CLASS = MilvusProxy

print(f"Extending from {SUPER_CLASS}")

class MilvusIndexProcessor(SUPER_CLASS):
    async def localConfigure(self):
        pass

    def get_default_arguments(self):
        return {
            'indexintent': False,
            'indexsql': True
        }
    
    async def index_qa(self, args, default_arguments):
        named_inputs = args['namedInputs']
        knowledge = named_inputs['knowledge']
        database = named_inputs['database']
        collection = named_inputs['collection']

        self.use_database(database, False)
        # Maybe the next line us useless...
        #await self.wait_loaded(collection)

        only_intexed_text = []
        for item in knowledge:
            only_intexed_text.append(item['text_indexed'])

        # Compute all embbeds at once
        embbeds = bge_m3_ef.encode_queries(only_intexed_text)

        for i in range(len(knowledge)):
            item = knowledge[i]
            item['id'] = f"{uuid.uuid4()}"
            item['sparse_vector'] = self.csr_to_dict(embbeds['sparse'][i])
            item['dense_vector'] = embbeds['dense'][i].tolist()
        # Insert all at once
        self.insert(knowledge, collection)

        return {
            'indexed': len(knowledge)
        }

    def csr_to_dict(self, sparse_csr):
        """Convert a scipy CSR sparse vector to a dictionary {index: value}."""
        coo = sparse_csr.tocoo()  # Convert to COOrdinate format
        return {int(i): float(v) for i, v in zip(coo.col, coo.data)}  # Extract nonzero values


    async def search_qa(self, args, default_arguments):
        named_inputs = args['namedInputs']
        query = named_inputs['query']
        database = named_inputs['database']
        collection = named_inputs['collection']
        k = named_inputs['kReRank']
        output_fields = [
            'id', 
            'document_id', 
            'text_indexed', 
            'text_answer', 
            'sparse_vector', 
            'dense_vector'
            ]

        self.use_database(database, False)
        embbeds = bge_m3_ef.encode_queries([query])
        milvus_client = MilvusHandler.get_client()

        query_dense = embbeds['dense'][0].tolist()
        query_sparse = self.csr_to_dict(embbeds['sparse'][0])

        def retrieve_top_k(query_dense, query_sparse, top_k, output_fields, use_dense = True):
            search_params = {
                "metric_type": "L2" if use_dense else "IP", 
                "params": {"nprobe": 10}
                }
            results = milvus_client.search(
                collection,
                data=[query_dense if use_dense else query_sparse],
                anns_field="dense_vector" if use_dense else "sparse_vector",
                search_params=search_params,
                limit=top_k,
                output_fields=output_fields,
                consistency_level="Strong"
            )
            result_list = []
            for result in results:
                for item in result:
                    result_list.append(item)
            return result_list
        
        def rerank_results(query_dense, query_sparse, retrieved_data, alpha=0.5, use_dense = False):
            scores = []
            
            reference = query_dense if use_dense else query_sparse

            for item in retrieved_data:
                doc_id = item["id"]
                other_score = item["distance"]
                this_vector = item['entity'].get("dense_vector", {}) if use_dense else item['entity'].get("sparse_vector", {})

                # Compute sparse similarity (dot product with binary mask approach)
                sparse_score = sum(reference.get(k, 0) * this_vector.get(k, 0) for k in reference.keys())

                # Normalize scores
                other_score = 1 / (1 + other_score)  # Convert L2 distance to similarity
                sparse_score = sparse_score / (sum(reference.values()) + 1e-8)  # Normalize by query weight

                # Weighted fusion score
                final_score = alpha * other_score + (1 - alpha) * sparse_score
                scores.append((doc_id, final_score))

            # Sort by final score (higher is better)
            scores.sort(key=lambda x: x[1], reverse=True)
            return scores

        result = retrieve_top_k(query_dense, query_sparse, k, output_fields, True)
        scores = rerank_results(query_dense, query_sparse, result, 0.5, False)

        # get the first
        idResult = scores[0][0]
        #print(idResult)
        most_relevant = list(filter(lambda item: item['id'] == idResult, result))
        #print(most_relevant)

        def simplify_element(input):
            return {
                "id": input['id'],
                "distance": input['distance'],
                "document_id": input['entity']['document_id'],
                "text_indexed": input['entity']['text_indexed'],
                "text_answer": input['entity']['text_answer'],
            }

        return {
            "no_rerank": simplify_element(result[0]),
            "rerank": simplify_element(most_relevant[0]),
        }

    async def delete_qa(self, args, default_arguments):
        named_inputs = args['namedInputs']
        item = named_inputs['item']
        database = named_inputs['database']
        collection_name = named_inputs['collection']

        milvus_client = MilvusHandler.get_client()
        self.use_database(database, False)
        expresion = f"id in [{item['id']}]"
        milvus_client.delete(collection_name=collection_name, filter=expresion)

        #connections.connect(alias=database, host="milvus", port="19530")
        #collection = Collection(collection_name)
        # Perform compaction
        #collection.compact()

        return {}
    
    async def update_qa(self, args, default_arguments):
        named_inputs = args['namedInputs']
        item = named_inputs['item']
        database = named_inputs['database']
        collection = named_inputs['collection']

        milvus_client = MilvusHandler.get_client()
        self.use_database(database, False)

        embbeds = bge_m3_ef.encode_queries([item['text_indexed']])
        item['sparse_vector'] = self.csr_to_dict(embbeds['sparse'][0])
        item['dense_vector'] = embbeds['dense'][0].tolist()

        print(f"Upsert into {database} and {collection}")

        milvus_client.upsert(
            collection_name=collection,
            data=[{
                #"id": np.int64(int(item['id'])),
                "id": item['id'],
                "document_id": item['document_id'],
                "text_indexed": item['text_indexed'],
                "text_answer": item['text_answer'],
                "sparse_vector": item['sparse_vector'],
                "dense_vector": item['dense_vector']
            }]
        )
        return {}
    
    async def pgtest(self, args, default_arguments):
        conn_params = {
            'user': os.environ['POSTGRES_USER'],
            'password': os.environ['POSTGRES_PASSWORD'],
            'database': os.environ['POSTGRES_DB'],
            'host': os.environ['POSTGRES_HOST'], 
            'port': int(os.environ['POSTGRES_PORT'])
        }

        # Establish a connection
        conn = await asyncpg.connect(**conn_params)

        #await conn.execute('''INSERT INTO users(name, dob) VALUES($1)''', 
        #                   'Bob')

        row = await conn.fetchrow(
        "SELECT NOW(), $1 as text", 'Bob')

        print(row)

        await conn.close()
        return {}
    
    async def process(self, args, default_arguments):
        method = args['method']
        if method == "indexqa":
            return await self.index_qa(args, default_arguments)
        elif method == "searchqa":
            return await self.search_qa(args, default_arguments)
        elif method == "deleteqa":
            return await self.delete_qa(args, default_arguments)
        elif method == "updateqa":
            return await self.update_qa(args, default_arguments)
        elif method == "pgtest":
            return await self.pgtest(args, default_arguments)

        return {}


async def main():
    processor = MilvusIndexProcessor()
    await processor.configure(sys.argv)
    await processor.start_communication()

if __name__ == '__main__':
    asyncio.run(main())
