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
    model, AnnSearchRequest, RRFRanker, Collection
)
from base_procesor import register_intent_found
import gc
import torch
import numpy as np

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

    def embed_all_MiniLM_L6_v2(self, senteces_list):
        return ef.encode_documents(senteces_list)

    def embed_bge_m3_ef_dense(self, senteces_list):
        print("embed_bge_m3_ef_dense")
        docs_embeddings = bge_m3_ef.encode_documents(senteces_list)
        return docs_embeddings["dense"]

    def embed_query_bge_m3_ef_dense(self, senteces_list):
        print("embed_bge_m3_ef_dense")
        query_embeddings = bge_m3_ef.encode_queries(senteces_list)
        return query_embeddings["dense"]

    async def index_faqs(self, args, default_arguments):
        data = args['data']
        db_name = data['db']
        collection_name = data['collection']
        named_inputs = args['namedInputs']

        self.use_database(db_name, False)
        await self.wait_loaded(collection_name)

        all_lines = ""

        if "text" in named_inputs:
            all_lines = named_inputs["text"].splitlines()
        elif "file" in data:
            path = data['file']
            f = open(path)
            all_lines = f.readlines()

        if len(all_lines) == 0:
            return {}

        current_question = None
        current_answer = None

        pattern = re.compile("^\s*¿")

        def clean_sentence(sentence):
            return re.sub(r'[\n\r]', '', sentence)

        current_count = 0

        def save_and_index_faq(one_question, one_answer):
            item = {
                'query': clean_sentence(one_question)[:1024],
                'answer': clean_sentence(one_answer)[:1024],
            }
            # Convert query to vector
            query_embeddings = bge_m3_ef.encode_queries([item['query']])
            item['sparse_vector'] = query_embeddings["sparse"][0]
            item['dense_vector'] = query_embeddings["dense"][0]
            self.insert(item, collection_name)

        for some_line in all_lines:
            if pattern.match(some_line):
                if current_question is not None and current_answer is not None:
                    save_and_index_faq(current_question, current_answer)
                    current_count = current_count + 1
                current_question = some_line
                current_answer = None
            else:
                if current_question is not None:
                    if current_answer is None:
                        current_answer = some_line
                    else:
                        current_answer = current_answer + some_line

        if current_question is not None and current_answer is not None:
            save_and_index_faq(current_question, current_answer)
            current_count = current_count + 1

        count = self.count(collection_name)
        await self.wait_release(collection_name)
        return {
            "count": count,
            "currentCount": current_count
        }

    async def general_search(self, db_name, collection_name, query, k, output_fields, query_field_name):
        client = MilvusHandler.client
        self.use_database(db_name, False)
        await self.wait_loaded(collection_name)

        #print(f"Searchig {query}")
        query_embeddings = bge_m3_ef.encode_queries([query])

        # Prepare the search requests for both vector fields
        sparse_search_params = {"metric_type": "IP"}
        sparse_req = AnnSearchRequest(query_embeddings["sparse"],
                                      "sparse_vector", sparse_search_params, limit=k)
        dense_search_params = {"metric_type": "IP"}
        dense_req = AnnSearchRequest(self.toDtypeFloat32(query_embeddings["dense"]),
                                     "dense_vector", dense_search_params, limit=k)

        # Search topK docs based on dense and sparse vectors and rerank with RRF.
        col = Collection(collection_name, using=client._using)
        res = col.hybrid_search([sparse_req, dense_req], rerank=RRFRanker(),
                                limit=k, output_fields=output_fields)
        res = res[0]
        result_texts = [hit.fields[query_field_name] for hit in res]
        #bge_rf = BGERerankFunction(device=f"{device}")
        bge_rf = BGERerankFunction(device="cpu")
        results = bge_rf(query, result_texts, top_k=k)
        the_response = []

        for hit in results:
            def search_hit_by_query(one_result):
                return one_result.fields[query_field_name] == hit.text
            filtered_result = list(filter(search_hit_by_query, res))
            if len(filtered_result) > 0:
                sinle_result = filtered_result[0]
                res_item = {
                    "score": hit.score
                }
                for field_name in output_fields:
                    res_item[field_name] = sinle_result.fields[field_name]
                the_response.append(res_item)
        return the_response

    async def search_faq(self, args, default_arguments):
        named_inputs = args['namedInputs']
        data = args['data']
        query = named_inputs['query']
        db_name = named_inputs['db']
        collection_name = named_inputs['collection']
        k = int(named_inputs['k'])
        output_fields = ['query', 'answer']
        query_field_name = 'query'
        return await self.general_search(
            db_name,
            collection_name,
            query,
            k,
            output_fields,
            query_field_name
        )

    async def search_intent(self, args, default_arguments):
        print("search_intent")
        data = args['data']
        db_name = data['db']
        collection_name = data['collection']
        named_inputs = args['namedInputs']
        output_fields = ["document_id", "text",
                         "speaker", "start_time", "end_time", 'ref_id']
        query_field_name = "text"
        return await self.general_search(
            db_name,
            collection_name,
            named_inputs['query'],
            named_inputs['k'],
            output_fields,
            query_field_name
        )

    async def index_intent(self, args, default_arguments):
        data = args['data']#ok
        room = args['room']#ok
        # Milvus params
        milvus_db_name = data['db']#ok
        milvus_collection_name = data['collection']#ok
        # Postgres params accountId, appId, mediaId, mediaType
        sql_detail = None
        if 'dbData' in args:
            sql_detail = args['dbData']
        # media, diarization
        named_inputs = args['namedInputs']
        # startTime
        media = named_inputs['media']
        # The diarization
        diarization = named_inputs['diarization']
        partial = []
        if ('partial' in diarization):
            partial = diarization['partial']

        print(default_arguments)
        
        if len(partial) == 0:
            return {}

        if default_arguments['indexintent']:
            self.use_database(milvus_db_name, False)
            
        for local_item in partial:
            # Gater information
            text = local_item['text']
            speaker = local_item['speaker']
            start_time_ms = int(local_item['start_time']*1000 + media['startTime'])
            end_time_ms = int(local_item['end_time']*1000 + media['startTime'])
            sql_inserted_id = ''
        
            # Postgres indexing
            if default_arguments['indexsql'] and sql_detail is not None:
                sql_insert_response = await register_intent_found(sql_detail, {
                    'mediaStartTime': start_time_ms,
                    'mediaEndTime': end_time_ms,
                    'mediaSourceUrl': '',
                    'text': text,
                    'textScore': 0,
                    "speaker": speaker,
                })
                if (sql_insert_response is not None and sql_insert_response['inserted'] >= 0):
                    sql_inserted_id = sql_insert_response['object']['id']
                
            # Milvus indexing
            if default_arguments['indexintent']:
                docs_embeddings = bge_m3_ef.encode_documents([text])
                item = {
                    "text": text,
                    "document_id": room,
                    "speaker": speaker,
                    "start_time": start_time_ms,
                    "end_time": end_time_ms,
                    'ref_id': sql_inserted_id
                }
                dense = docs_embeddings["dense"][0] # this is a numpy.ndarray!!
                item['dense_vector'] = self.toDtypeFloat32(dense)
                sparse = docs_embeddings["sparse"][0] # this is a scipy.sparse._csr.csr_matrix!!
                item['sparse_vector'] = sparse

                if issubclass(self.__class__, MilvusProxy):
                    # MilvusProxy needs change in the arrays
                    item['dense_vector'] = self.numpy_array_to_native_array(item['dense_vector'])
                    item['sparse_vector'] = self.numpy_array_to_native_array(item['sparse_vector'].toarray()[0])

                self.insert(item, milvus_collection_name)

        return {
        }

    async def compare(self, args, default_arguments):
        named_inputs = args['namedInputs']
        phrase1 = named_inputs['phrase1']
        phrases = named_inputs['phrase2'].splitlines()
        phrases.insert(0, phrase1)
        docs_embeddings = bge_m3_ef.encode_documents(phrases)['dense']
        
        tasks = []
        for i in range(1, len(phrases)):
            tasks.append(self.compareTwo(docs_embeddings[0], docs_embeddings[i]))
        
        distance = await asyncio.gather(*tasks)
        return {
            'distance': distance
        }
        
    async def process(self, args, default_arguments):
        method = args['method']
        if method == "indexintent":
            return await self.index_intent(args, default_arguments)
        elif method == "searchintent":
            return await self.search_intent(args, default_arguments)
        elif method == "indexfaqs":
            return await self.index_faqs(args, default_arguments)
        elif method == "searchfaq":
            return await self.search_faq(args, default_arguments)
        elif method == "compare":
            return await self.compare(args, default_arguments)

        return {}


async def main():
    processor = MilvusIndexProcessor()
    await processor.configure(sys.argv)
    await processor.start_communication()

if __name__ == '__main__':
    asyncio.run(main())
