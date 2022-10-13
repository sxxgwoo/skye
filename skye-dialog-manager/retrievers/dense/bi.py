# python
import os
import logging
import json
# 3rd-party
import numpy as np
import faiss

# framework
from retrievers.base import DenseRetriever
from models.encoder.bi import BiEncoderModel


logger = logging.getLogger(__name__)


class BiEncoderDenseModel(DenseRetriever):
    def __init__(self, 
                 encoder: BiEncoderModel,
                 data: dict):
        self.encoder = encoder
        self.read_index(data)
        
    def read_index(self, data: dict):
        db = {}
        for persona, files in data['files'].items():
            index_filepath = os.path.join(data['dir'], files['index'])
            index = faiss.read_index(index_filepath)
            db[persona] = index
        self.db = db
        
    def inference(self, persona: str, query: str, k=5):
        if persona not in self.db.keys():
            raise NameError("{0} persona is not supported. supported list: [{1}]".format(persona, self.db.keys()))
        dialog_input = query
        embedded_query = self.encoder.encode([dialog_input])
        faiss.normalize_L2(embedded_query)
        D, I = self.db[persona].search(embedded_query, k=k)
        return D[0], I[0]


