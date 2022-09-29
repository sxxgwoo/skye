# python
from typing import Dict, List
from dataclasses import asdict
from contextvars import ContextVar
import logging
from copy import deepcopy
import os
import json
import random
import defusedxml

# 3rd-party
import faiss
import numpy as np
from hydra.utils import to_absolute_path, instantiate

# framework
from data import STSRetrieveItem, STSTime, STSRetrieveResult
from decorator import time_usage
from load_data import load_sentiment
# from ans_para import get_para

logger = logging.getLogger(__name__)

CONFIG = ContextVar('config', default=None)
DB = ContextVar('db', default=None)
DENSE_RETRIEVER = ContextVar('dense_retriever', default=None)
SENTI_DB = ContextVar('senti_db', default=None)


def sts_init(app):
    # load config
    config = app['config']

    # load encoder
    logger.info(f"Instantiating encoder <{config.model.encoder._target_}>")
    encoder = instantiate(config.model.encoder)

    # unpack db
    logger.info('Unpack DB')
    db = _unpack_db(config.data)

    #unpack senti_db
    logger.info('Unpack senti_db')
    # senti_db = _unpack_senti_db(config.sentiment)

    # load dense retriever
    dense_kwargs = {'encoder':encoder, 'data': config.data}
    logger.info(f"Instantiating dense retriever <{config.retriever.dense._target_}>")
    dense_retriever = instantiate(config.retriever.dense, **dense_kwargs)
    
    # set context var
    CONFIG.set(config)
    DB.set(db)
    # SENTI_DB.set(senti_db)
    DENSE_RETRIEVER.set(dense_retriever)



def _unpack_db(data: dict):
    db = {}
    for persona, files in data['files'].items():
        logger.info('read persona: {0}'.format(persona))
        qna_filepath = os.path.join(data['dir'], files['qna'])
        answers_filepath = os.path.join(data['dir'], files['answers'])
        with open(qna_filepath,encoding='utf-8-sig') as fp:
            qna = json.load(fp)
        with open(answers_filepath,encoding='utf-8-sig') as fp:
            answers = json.load(fp)
        db[persona] = {
            'qna': qna,
            'answers': answers
        }
    return db

# def _unpack_senti_db(data: dict):
#     db = {}
#     for persona, files in data['files'].items():
#         logger.info('read persona: {0}'.format(persona))
#         answers_filepath = os.path.join(data['dir'], files['answers'])
#         with open(answers_filepath) as fp:
#             answers = json.load(fp)
#         db[persona] = {
#             'answers': answers
#         }
#     return db



@time_usage
def run(user, *args, **kwargs) -> Dict:
    documents = retrieve(**kwargs['body'])
    result = STSRetrieveResult(documents=documents, msg='정상 처리되었습니다.')

    return asdict(result)

def para(persona, answer):
    db = DB.get()
    if answer not in db[persona]['answers'].keys():
        return answer
    return random.sample(db[persona]['answers'][answer], k=1)[0]

def retrieve(persona: str, query: str, k: int = 5, score_threshold: int = 0.7, *args, **kwargs) -> List[STSRetrieveItem]:
    db = DB.get()
    dense_retriever = DENSE_RETRIEVER.get()

    D, I = dense_retriever.inference(persona, query, k)
    documents = [STSRetrieveItem(
        score=float(distance),
        id=db[persona]['qna']['data'][idx]['id'],
        question=db[persona]['qna']['data'][idx]['question'],
        answer=para(persona, db[persona]['qna']['data'][idx]['answer'])    # answer = para(persona,paraphrasing된 데이터에서 index에 해당하는 asnwer)
        # # answer=random.sample(get_para(db[persona]['qna']['data'][idx]['answer']),k=1)[0]    # 직접 pegasus모델 사용하여 answer를 바로 paraphrasing해보기
        # answer = db[persona]['qna']['data'][idx]['answer']      #paraphrasing하지 않는 경우
    ) for distance, idx in zip(D, I) if distance > score_threshold]

    return documents

def longterm_retrieve(query: str, k: int = 5, score_threshold: int = 0.7, *args, **kwargs) -> List[STSRetrieveItem]:
    db = DB.get()
    dense_retriever = DENSE_RETRIEVER.get()

    D, I = dense_retriever.inference('user_longterm', query, k)
    documents = [STSRetrieveItem(
        score=float(distance),
        id=db['user_longterm']['qna']['data'][idx]['id'],
        question=db['user_longterm']['qna']['data'][idx]['question'],
        answer=para('user_longterm', db['user_longterm']['qna']['data'][idx]['answer'])    # answer = para(persona,paraphrasing된 데이터에서 index에 해당하는 asnwer)

    ) for distance, idx in zip(D, I) if distance > score_threshold]

    return documents

def time_retrieve(query: str, k: int =1, score_threshold: int = 0.7, *args, **kwargs) -> List[STSRetrieveItem]:
    db = DB.get()
    dense_retrieve = DENSE_RETRIEVER.get()

    D, I = dense_retrieve.inference('question_time', query, k)
    documents = [STSTime(
        score=float(distance),
        time_zone=db['question_time']['qna']['data'][idx]['time zone'],
        region=db['question_time']['qna']['data'][idx]['region'],
        question=db['question_time']['qna']['data'][idx]['question'],
        answer_type=db['question_time']['qna']['data'][idx]['answer']

    ) for distance, idx in zip(D,I) if distance > score_threshold]

    return documents

def weather_retreive(query:str, k: int =1, score_threshold: int = 0.7, *args, **kwargs):
    db = DB.get()
    dense_retrieve = DENSE_RETRIEVER.get()

    D, I = dense_retrieve.inference('question_weather', query, k)
    documents = [STSRetrieveItem(
        score=float(distance),
        id=db['question_weather']['qna']['data'][idx]['region'],
        question=db['question_weather']['qna']['data'][idx]['question'],
        answer=para('question_weather',db['question_weather']['qna']['data'][idx]['answer'])
    ) for distance, idx in zip(D,I) if distance > score_threshold
    ]

    return documents


def question_generation_rule(user_name: str):
    # senti_db = SENTI_DB.get()
    senti_chat = load_sentiment(user_name)
    # response_test = ''
    
    if len(senti_chat) >= 4:
        if senti_chat[str(len(senti_chat)-1)][0]['skye'][-1] != '?'\
            and senti_chat[str(len(senti_chat)-2)][0]['skye'][-1] != '?'\
            and senti_chat[str(len(senti_chat)-3)][0]['skye'][-1] != '?':

            return True





    
    
    

