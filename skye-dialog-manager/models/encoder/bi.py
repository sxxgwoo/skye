# python
from typing import Any, List, Union

# 3rd-party
import torch
from torch import nn, Tensor
from torch.optim import AdamW
import numpy as np
from numpy import ndarray
import pytorch_lightning as pl
import tqdm


class BiEncoderModel():
    def __init__(self, 
        model = None,
        tokenizer = None,                   # transformer 모델과 tokenizer를 paramter로 받음
        device = 'cuda'
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
    
    def encode(self, sentences: Union[str, List[str], List[int]],
               batch_size: int = 32,
               show_progress_bar: bool = None,
               output_value: str = 'sentence_embedding',
               convert_to_numpy: bool = True,
               convert_to_tensor: bool = False
            ) -> Union[List[Tensor], ndarray, Tensor]:
        """
        Computes sentence embeddings
        :param sentences: the sentences to embed
        :param batch_size: the batch size used for the computation
        :param show_progress_bar: Output a progress bar when encode sentences
        :param output_value:  Default sentence_embedding, to get sentence embeddings. Can be set to token_embeddings to get wordpiece token embeddings.
        :param convert_to_numpy: If true, the output is a list of numpy vectors. Else, it is a list of pytorch tensors.
        :param convert_to_tensor: If true, you get one large tensor as return. Overwrites any setting from convert_to_numpy
        :param device: Which torch.device to use for the computation
        :param normalize_embeddings: If set to true, returned vectors will have length 1. In that case, the faster dot-product (util.dot_score) instead of cosine similarity can be used.
        :return:
           By default, a list of tensors is returned. If convert_to_tensor, a stacked tensor is returned. If convert_to_numpy, a numpy matrix is returned.
        """
        if convert_to_tensor:
            convert_to_numpy = False

        if output_value == 'token_embeddings':
            convert_to_tensor = False
            convert_to_numpy = False

        input_was_string = False
        if isinstance(sentences, str) or not hasattr(sentences, '__len__'): #Cast an individual sentence to a list with length 1
            sentences = [sentences]
            input_was_string = True

        all_embeddings = []
        length_sorted_idx = np.argsort([-len(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        batch_loop = range(0, len(sentences), batch_size)
        if show_progress_bar:
            batch_loop = tqdm.tqdm(batch_loop)

        for start_index in batch_loop:
            sentences_batch = sentences_sorted[start_index:start_index+batch_size]
            features = self.tokenizer.batch_encode_plus(sentences_batch, padding=True, truncation=True, return_tensors='pt')
            # features = batch_to_device(features, device)
            features = {k: v.to(self.device) for k, v in features.items()}

            with torch.no_grad():
                embeddings = self.model(**features, output_hidden_states=True, return_dict=True).pooler_output

                all_embeddings.extend(embeddings)

        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]

        if convert_to_tensor:
            all_embeddings = torch.stack(all_embeddings)
        elif convert_to_numpy:
            all_embeddings = np.asarray([emb.cpu().numpy() for emb in all_embeddings])

        if input_was_string:
            all_embeddings = all_embeddings[0]

        return all_embeddings