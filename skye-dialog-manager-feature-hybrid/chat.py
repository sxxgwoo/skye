# python
from copy import deepcopy
from typing import Dict
from dataclasses import asdict
import logging
from enum import IntEnum
import random
import json
import os

from requests import Response

# framework
from data import HybridResult
from decorator import time_usage
from sts import longterm_retrieve, retrieve, time_retrieve, question_generation_rule , weather_retreive
# from peep_talk import run as generation_run
from openAI import run as generation_run
from save_sentiment import retrieve_senti, save_extended
from load_data import load_answer
from tool import paraphrasing_question, present_time , present_weather
from analysis_class import sentiment_analysis_class as sentiment_analysis
from analysis_class import topic_extraction_class as topic_extraction
from analysis_class import question_generation_class as question_generation
from analysis_class import topic_word_class as topic_word
from analysis_class import emotion_analysis_class as emotion_analysis


logger = logging.getLogger(__name__)


class ResponseType(IntEnum):
    RETRIEVE = 0
    GENERATION = 1

@time_usage
def run(user, *args, **kwargs) -> Dict:
    # run retrieve
    retrieve_threshold = kwargs['body']['retrieve_threshold']
    documents_longterm = longterm_retrieve(**kwargs['body'])
    documents = retrieve(**kwargs['body'])
    documents_time = time_retrieve(**kwargs['body'])
    documents_weather = weather_retreive(**kwargs['body'])

    retrieve_top_score_longterm = max([0] + [doc.score for doc in documents_longterm])
    retrieve_top_score = max([0] + [doc.score for doc in documents])
    retrieve_top_score_time = max([0] + [doc.score for doc in documents_time])
    retrieve_top_score_weather = max([0] + [doc.score for doc in documents_weather])
    name = kwargs['body']['name']

    user_sentiment = sentiment_analysis(kwargs['body']['query'])    # dictionary 형태의 감성분석 결과
    user_sentiment_temp = deepcopy(user_sentiment)
    topic = topic_extraction(kwargs['body']['query'])               # 토픽 결과
    word = topic_word(kwargs['body']['query'])               # 토픽 결과
    emotion = emotion_analysis(kwargs['body']['query'])               # 토픽 결과
    

    # check retrieve threshold
    if retrieve_top_score_longterm >= 0.85:
        bot_response_type = ResponseType.RETRIEVE
        bot_response = documents_longterm[0].answer               # longterm memory에서 retrieve
        documents = documents_longterm

    elif retrieve_top_score_time > 0.85:
        bot_response_type = ResponseType.RETRIEVE
        time_zone = documents_time[0].time_zone                                # 현재 시간 retrieve
        answer_type = documents_time[0].answer_type
        region = documents_time[0].region
        if time_zone == 'korner':
            bot_response = documents_time[0].answer_type
        else:
            bot_response = present_time(time_zone, answer_type,region)
        documents = documents_time

    elif retrieve_top_score_weather > 0.85:
        bot_response_type = ResponseType.RETRIEVE
        region = documents_weather[0].id
        if region == 'korner':
            bot_response = documents_weather[0].answer
        else:
            bot_response = present_weather(region)         # 현재 날씨 retrieve
        documents = documents_weather

    
    elif retrieve_top_score < retrieve_threshold or ("what" in kwargs['body']['query'] and "my" in kwargs['body']['query']): 
        bot_response_type = ResponseType.GENERATION
        
        if topic =="" or word =="":         # topic 이나 word가 추출되지 않을때 pre condition 
            pre =""
        else:
            if os.path.exists('/home/dmlab/skye/skye-dialog-manager/data/userinfo/data_'+name+'.json'):
                with open('/home/dmlab/skye/skye-dialog-manager/data/userinfo/data_'+name+'.json','r',encoding='utf-8-sig') as json_file:
                    json_data = json.load(json_file)
                pre_con=[]
                tr = len(json_data)-1
                for i in range(1,tr):
                    if json_data[str(i)][0]["topic"]==topic:
                        if json_data[str(i)][0]["pre"] =='':
                            pass
                        else:
                            pre_con += json_data[str(i)][0]["pre"]
                    else:
                        pass
            else:
                pre_con=""    
            pre = ','.join(pre_con)+"."

        generation_kwargs = {'body': 
            {
            'text': kwargs['body']['query'],
            'name': kwargs['body']['name'],
            'pre': pre
            }
        }
        bot_response = generation_run(**generation_kwargs).strip()
    
    else:
        bot_response_type = ResponseType.RETRIEVE                 # 기존 persona retrieve
        bot_response = documents[0].answer

    k=[]        # emotion 가장 높은 score 값 추출
    for i in range(7):

        k.append(emotion[0][i]["score"])
        tmp= max(k)
        index = k.index(tmp)

    user_sentiment[0]['user'] = kwargs['body']['query']             # 각 대화 저장
    user_sentiment[0]['skye'] = bot_response
    user_sentiment[0]['topic'] = topic                              # 토픽 저장
    user_sentiment[0]['word'] = word                                # 관련 토픽 단어 저장
    user_sentiment[0]['emotion'] = emotion                                # 감정 저장
    if topic !="" and word !="" and emotion[0][index]["score"]-emotion[0][4]["score"] > 0.5:
        user_sentiment[0]['pre'] = 'Human'+" "+ word +" "+ emotion[0][index]["label"]
    else:
        user_sentiment[0]['pre'] =''
    retrieve_senti(name,user_sentiment)
    
    highest_senti = max(user_sentiment_temp[0],key=user_sentiment_temp[0].get)     # 가장 확률값이 높은 sentiment

    if '?' in bot_response or '?' in kwargs['body']['query']:
        extended_answer = ""
    
    elif topic and question_generation_rule(name) == True:
        answer = random.choice(list(load_answer()[topic][highest_senti].values()))                                     # answer 뽑기
        extended_answer = question_generation(answer,kwargs['body']['query']).strip("<pad> question:""</s>")              #question generation
        extended_answer = " "+paraphrasing_question(extended_answer)                                                          # me -> you 등등
    else:
        extended_answer = ""

    result = save_extended(name,extended_answer)
    if result == True:
        extended_answer = ""


    # make result
    result = HybridResult(bot_response_type = bot_response_type,
                          bot_response = bot_response+extended_answer,
                          sts_result = documents,
                          msg = '정상 처리되었습니다.')

    

    return asdict(result)


