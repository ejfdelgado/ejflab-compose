from base_procesor import BaseProcessor
import os
import json
import numpy as np

class MilvusCommon(BaseProcessor):
    def toDtypeFloat32(self, input):
        return np.array([x for x in input], dtype=np.float32)
    
    async def compareTwo(self, docs_embeddings1, docs_embeddings2):
        distance = self.cosine_distance(docs_embeddings1, docs_embeddings2).item()
        return distance
    
    def cosine_distance(self, ar1, ar2):
        A = np.array(ar1)
        B = np.array(ar2)
        dot_product = np.dot(A, B)
        magnitude_A = np.linalg.norm(A)
        magnitude_B = np.linalg.norm(B)
        cosine_similarity = dot_product / (magnitude_A * magnitude_B)
        return cosine_similarity