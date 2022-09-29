from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelWithLMHead, AutoModel, pipeline
import numpy as np
from scipy.special import softmax,expit
import torch

import time

import torch
from scipy.spatial.distance import cosine

import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

import spacy


class analysis_all:

    def __init__(self):
        
        torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # MODEL_SENTI = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
        # self.tokenizer_senti = AutoTokenizer.from_pretrained(MODEL_SENTI)
        # self.model_senti = AutoModelForSequenceClassification.from_pretrained(MODEL_SENTI)

        MODEL_TOPIC = f"cardiffnlp/tweet-topic-19-multi"
        self.tokenizer_topic = AutoTokenizer.from_pretrained(MODEL_TOPIC)
        self.model_topic = AutoModelForSequenceClassification.from_pretrained(MODEL_TOPIC)
        self.class_mapping = self.model_topic.config.id2label

        # MODEL_QUESTION = f"mrm8488/t5-base-finetuned-question-generation-ap"
        # self.tokenizer_question = AutoTokenizer.from_pretrained(MODEL_QUESTION)
        # self.model_question = AutoModelWithLMHead.from_pretrained(MODEL_QUESTION)

        # MODEL_WORD = f"princeton-nlp/sup-simcse-bert-base-uncased"
        # self.tokenizer_word = AutoTokenizer.from_pretrained(MODEL_WORD)
        # self.model_word = AutoModel.from_pretrained(MODEL_WORD)

        # MODEL_EMOTION = f"j-hartmann/emotion-english-distilroberta-base"
        # self._classifier = pipeline("text-classification", model=MODEL_EMOTION, return_all_scores=True)


    def preprocess(self,text):
        new_text = []
 
        for t in text.split(" "):
            t = '@user' if t.startswith('@') and len(t) > 1 else t
            t = 'http' if t.startswith('http') else t
            new_text.append(t)
        return " ".join(new_text)
    
    def sentiment_analysis(self,sentence):
        labels = ['negative','neutral','positive']
        text = self.preprocess(sentence)
        encoded_input = self.tokenizer_senti(text, return_tensors='pt')
        output = self.model_senti(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        
        ranking = np.argsort(scores)
        ranking = ranking[::-1]
        
        data = []
        temp_dict = []
        for i in range(scores.shape[0]):
            l = labels[ranking[i]]
            s = scores[ranking[i]].item()
            temp_dict.append({f'{l}':s})
        merged = {**temp_dict[0],**temp_dict[1],**temp_dict[2]}
        data.append(merged)
        
        return data
    

    def topic_extraction(self,text):
        tokens = self.tokenizer_topic(text, return_tensors = 'pt')
        output = self.model_topic(**tokens)
        
        topics = []
        scores = output[0][0].detach().numpy()
        scores = expit(scores)
        # if np.max(scores) >= 0.65:
        #     predictions = (scores == np.full((1,19),np.max(scores))) * 1
        # else:
        #     predictions = np.full((1,19),0)
        predictions =(np.where((scores>=0.65) & (scores == np.max(scores))))*1
        for i in range(len(predictions)):
            if predictions[i]:
                topics.append(self.class_mapping[i])
        topic = ','.join(topics) 
    
        return topic
    
    def question_generation(self,answer, context, max_length=64):
        input_text = "answer: %s  context: %s </s>" % (answer, context)
        features = self.tokenizer_question([input_text], return_tensors='pt')

        output = self.model_question.generate(input_ids=features['input_ids'], 
                attention_mask=features['attention_mask'],
                max_length=max_length)

        return self.tokenizer_question.decode(output[0])

    # def topic_word(self,text,topic):
    #     # list_text = text.split()
    #     texttemp =[]
    #     texttemp.append(topic)
    #     list_text = word_tokenize(text)
    #     pos = nltk.pos_tag(list_text)
    #     selective_pos = ['NN','NNS','NNP','NNPS']
    #     selective_pos_words = []
        
    #     for word,tag in pos:
    #         if tag in selective_pos:
    #             selective_pos_words.append(word)
    #     if len(selective_pos_words) >= 1 and topic !="":      #문장내 품사가 명사인것이 1개이상있을때 실행 
    #         texts=texttemp+selective_pos_words
    #         # for i in len(list_text):
    #         #     texts.append(list_text[i])
        
    #         inputs = self.tokenizer_word(texts, padding=True, truncation=True, return_tensors="pt")

    #         # Get the embeddings
    #         with torch.no_grad():
    #             embeddings = self.model_word(**inputs, output_hidden_states=True, return_dict=True).pooler_output
    #         cosine_sim=[]
    #         a= len(selective_pos_words)
    #         for i in range(a):
    #             cosine_sim.append(1-cosine(embeddings[0], embeddings[i+1]))
    #         if len(cosine_sim)>1:
    #             word = max(cosine_sim)
    #         else:
    #             word = cosine_sim
    #         index = cosine_sim.index(word)
    #         topicword=selective_pos_words[index]
    #     else:                               #문장내 품사가 명사인것이 없을때 
    #         topicword=""
    #     return topicword

    def word_spacy(self,text):
        nlp =spacy.load('en_core_web_sm')
        doc = nlp(text)
        selective_pos =['dobj','attr','ROOT','nsubj','pobj']
        selective_tag =['NN','NNP','NNS','NNPS','VBG']
        selective=''
        selective_pos_tag_word=[]
        for token in doc:
            if token.dep_ in selective_pos:
                if token.tag_ in selective_tag:
                    if token.dep_ == 'attr' and token.tag_ != 'VBG':
                        pass
                    else:
                        selective_pos_tag_word.append(token)
            else:
                pass
        return str(selective_pos_tag_word).replace('[','').replace(']','')



    def emotion_analysis(self, text):
 
        prediction = self._classifier(text)
        return prediction



def init():
    global analysis
    analysis = analysis_all()

# def sentiment_analysis_class(sentence):
#     return analysis.sentiment_analysis(sentence)

def topic_extraction_class(text):
    return analysis.topic_extraction(text)

# def question_generation_class(answer, context):
#     return analysis.question_generation(answer, context)

# def topic_word_class(text, topic):
#     return analysis.topic_word(text, topic)

# def topic_word_class(text):
#     return analysis.word_spacy(text)

# def emotion_analysis_class(text):
#     return analysis.emotion_analysis(text)


if __name__ == "__main__":

    print("Loading...")
    start = time.time()
    analysis_obj = analysis_all()
    end = time.time()
    print(f"It tooked : {end - start} seconds to load the model")
    print("model loaded!")
    # pred = analysis2_obj.topic_extraction("The U.S. Supreme Court’s decision Friday to end constitutional protection for abortion opened the gates for a wave of litigation.")
    
    # sent = analysis_obj.sentiment_analysis("I love coffee")
    # question = analysis_obj.question_generation("i want to play the soccer","Drawing is a form of visual art in which an artist uses instruments to mark paper or other two-dimensional surface")
    pred = analysis_obj.topic_extraction("I love coffee")
    # wo = analysis_obj.topic_word("i want to play the soccer",pred)
    # wo = analysis_obj.word_spacy("I love coffee")
    # emo = analysis_obj.emotion_analysis("I love coffee")
    # k=[]
    # for i in range(7):

    #     k.append(emo[0][i]["score"])
    #     tmp= max(k)
    #     index = k.index(tmp)

    # # pr = ','.join(pred)
    # print(sent)
    # print(question)
    print(pred)
    # print(type(wo))
    # print(wo)
    # print(emo)
    # print(emo[0][index]["score"])
    # print(emo[0][4]["score"])
    # print(emo[0][index]["score"]-emo[0][4]["score"])

    