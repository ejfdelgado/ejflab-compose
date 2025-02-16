import asyncio
import sys
import json
import os
from base_procesor import BaseProcessor
import asyncpg
import asyncio
import numpy as np
from scipy.sparse import coo_matrix
from transformers import AutoTokenizer, AutoModel
import torch
import traceback
import json
from scipy.sparse import coo_matrix

model_name = "BAAI/bge-m3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

class BaaiProcessor(BaseProcessor):
    async def localConfigure(self):
        pass

    def get_default_arguments(self):
        return {}

    def generate_vectors(self, text):
        # Tokenize input text
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)

        # Generate embeddings
        with torch.no_grad():
            outputs = model(**inputs)

        # Extract dense vector (CLS token embedding)
        dense_vector = outputs.last_hidden_state[:, 0, :].squeeze().numpy()

        # Extract sparse vector (e.g., using the model's sparse attention weights)
        # Note: BAAI/bge-m3 may not directly output sparse vectors, so this is a placeholder.
        # You may need to implement custom logic for sparse vector extraction.
        sparse_vector = coo_matrix(outputs.last_hidden_state.mean(dim=1).squeeze().numpy())

        return dense_vector, sparse_vector

    async def index(self, args, default_arguments):
        named_inputs = args['namedInputs']
        knowledge = named_inputs['knowledge']


        for i in range(len(knowledge)):
            item = knowledge[i]
            dense_vector, sparse_vector = self.generate_vectors(item['text_indexed'])
            item['sparse_vector'] = {
                'indices': sparse_vector.row.tolist(),
                'values': sparse_vector.data.tolist(),
                'shape': sparse_vector.shape
            }
            item['dense_vector'] = dense_vector.tolist()

        conn = await self.get_pg_connection()
        try:
            for i in range(len(knowledge)):
                item = knowledge[i]
                await conn.execute("""
                                   INSERT INTO public.knowledge
                                   (
                                    document_id,
                                    text_indexed,
                                    text_answer,
                                    dense_vector,
                                    sparse_vector
                                   )
                                   VALUES($1,$2,$3,$4,$5)
                                   """, 
                           item['document_id'],
                           item['text_indexed'],
                           item['text_answer'],
                           json.dumps(item['dense_vector']),
                           json.dumps(item['sparse_vector']),
                           )
        finally:
            await conn.close()

        return {
            'indexed': len(knowledge)
        }
    
    async def search(self, args, default_arguments):
        named_inputs = args['namedInputs']
        query = named_inputs['query']
        k = named_inputs['kReRank']

        query_dense, query_sparse = self.generate_vectors(query)
        conn = await self.get_pg_connection()
        reranked_results = []
        try:
            query_dense_list = query_dense.tolist()
            results = await conn.fetch(
                """
                SELECT
                    document_id,
                    text_indexed,
                    text_answer,
                    dense_vector,
                    sparse_vector,
                    1 - (dense_vector <=> $1) AS similarity
                FROM public.knowledge
                ORDER BY similarity DESC
                LIMIT $2
                """,
                json.dumps(query_dense_list), 
                k
            )
            
            for record in results:



                reranked_results.append({
                    "text_indexed": record['text_indexed']
                })
        finally:
            await conn.close()

        return {"rerank": reranked_results}

    async def delete(self, args, default_arguments):
        named_inputs = args['namedInputs']
        item = named_inputs['item']
        return {}

    async def update(self, args, default_arguments):
        named_inputs = args['namedInputs']
        item = named_inputs['item']
        return {}

    async def get_pg_connection(self):
        conn_params = {
            'user': os.environ['POSTGRES_USER'],
            'password': os.environ['POSTGRES_PASSWORD'],
            'database': os.environ['POSTGRES_DB'],
            'host': os.environ['POSTGRES_HOST'], 
            'port': int(os.environ['POSTGRES_PORT'])
        }

        # Establish a connection
        conn = await asyncpg.connect(**conn_params)
        return conn

    async def process(self, args, default_arguments):
        method = args['method']
        if method == "index":
            return await self.index(args, default_arguments)
        elif method == "search":
            return await self.search(args, default_arguments)
        elif method == "delete":
            return await self.delete(args, default_arguments)
        elif method == "update":
            return await self.update(args, default_arguments)


async def main():
    processor = BaaiProcessor()
    await processor.configure(sys.argv)
    await processor.start_communication()

if __name__ == '__main__':
    asyncio.run(main())
