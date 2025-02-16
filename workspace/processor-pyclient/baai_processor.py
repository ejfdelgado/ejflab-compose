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
    
    def csr_to_dict(self, sparse_csr):
        """Convert a scipy CSR sparse vector to a dictionary {index: value}."""
        coo = sparse_csr.tocoo()
        return {int(i): float(v) for i, v in zip(coo.col, coo.data)}

    async def search(self, args, default_arguments):
        named_inputs = args['namedInputs']
        query = named_inputs['query']
        k = named_inputs['kReRank']

        query_dense, query_sparse = self.generate_vectors(query)
        sparse_vectors_dicts = self.csr_to_dict(query_sparse)
        conn = await self.get_pg_connection()
        reranked_results = []
        try:
            query_dense_list = query_dense.tolist()
            results = await conn.fetch(
                """
                SELECT
                    id,
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
            
            alpha=0.5
            scores = []

            for item in results:
                id = item["id"]
                dense_score = 1 - item["similarity"]
                sparse_vector = json.loads(item["sparse_vector"])

                # Compute sparse similarity (dot product)
                sparse_score = sum(sparse_vectors_dicts.get(k, 0) * sparse_vector.get(k, 0) for k in sparse_vectors_dicts.keys())

                # Normalize scores
                dense_score = 1 / (1 + dense_score)  # Convert L2 distance to similarity
                sparse_score = sparse_score / (sum(sparse_vectors_dicts.values()) + 1e-8)  # Normalize

                # Weighted fusion score
                final_score = alpha * dense_score + (1 - alpha) * sparse_score
                scores.append((id, final_score))
                
            scores.sort(key=lambda x: x[1], reverse=True)
            idResult = scores[0][0]
            most_relevant = list(filter(lambda item: item['id'] == idResult, results))
        finally:
            await conn.close()

        def simplify_element(input):
            return {
                "id": input['id'],
                "distance": input['similarity'],
                "document_id": input['document_id'],
                "text_indexed": input['text_indexed'],
                "text_answer": input['text_answer'],
            }

        return {
            "no_rerank": simplify_element(results[0]),
            "rerank": simplify_element(most_relevant[0]),
        }

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
