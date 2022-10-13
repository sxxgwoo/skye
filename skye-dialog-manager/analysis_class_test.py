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



class analysis_all:

    def __init__(self):
        
        torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        MODEL_SENTI = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
        self.tokenizer_senti = AutoTokenizer.from_pretrained(MODEL_SENTI)
        self.model_senti = AutoModelForSequenceClassification.from_pretrained(MODEL_SENTI)

        MODEL_TOPIC = f"cardiffnlp/tweet-topic-19-multi"
        self.tokenizer_topic = AutoTokenizer.from_pretrained(MODEL_TOPIC)
        self.model_topic = AutoModelForSequenceClassification.from_pretrained(MODEL_TOPIC)
        self.class_mapping = self.model_topic.config.id2label

        MODEL_QUESTION = f"mrm8488/t5-base-finetuned-question-generation-ap"
        self.tokenizer_question = AutoTokenizer.from_pretrained(MODEL_QUESTION)
        self.model_question = AutoModelWithLMHead.from_pretrained(MODEL_QUESTION)

        MODEL_WORD = f"princeton-nlp/sup-simcse-bert-base-uncased"
        self.tokenizer_word = AutoTokenizer.from_pretrained(MODEL_WORD)
        self.model_word = AutoModel.from_pretrained(MODEL_WORD)

        MODEL_EMOTION = f"j-hartmann/emotion-english-distilroberta-base"
        self._classifier = pipeline("text-classification", model=MODEL_EMOTION, return_all_scores=True)


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
        predictions = (scores >= 0.65) * 1
        
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

    def topic_word(self,text,topic):
        # list_text = text.split()
        texttemp =[]
        texttemp.append(topic)
        list_text = word_tokenize(text)
        pos = nltk.pos_tag(list_text)
        selective_pos = ['NN','NNS','NNP','NNPS']
        selective_pos_words = []
        
        for word,tag in pos:
            if tag in selective_pos:
                selective_pos_words.append(word)
        if len(selective_pos_words) >= 1 and topic !="":      #문장내 품사가 명사인것이 1개이상있을때 실행 
            texts=texttemp+selective_pos_words
            # for i in len(list_text):
            #     texts.append(list_text[i])
        
            inputs = self.tokenizer_word(texts, padding=True, truncation=True, return_tensors="pt")

            # Get the embeddings
            with torch.no_grad():
                embeddings = self.model_word(**inputs, output_hidden_states=True, return_dict=True).pooler_output
            cosine_sim=[]
            a= len(selective_pos_words)
            for i in range(a):
                cosine_sim.append(1-cosine(embeddings[0], embeddings[i+1]))
            if len(cosine_sim)>1:
                word = max(cosine_sim)
            else:
                word = cosine_sim
            index = cosine_sim.index(word)
            topicword=selective_pos_words[index]
        else:                               #문장내 품사가 명사인것이 없을때 
            topicword=""
        return topicword

    def emotion_analysis(self, text):
 
        prediction = self._classifier(text)
        return prediction



def init():
    global analysis
    analysis = analysis_all()

def sentiment_analysis_class(sentence):
    return analysis.sentiment_analysis(sentence)

def topic_extraction_class(text):
    return analysis.topic_extraction(text)

def question_generation_class(answer, context):
    return analysis.question_generation(answer, context)

def topic_word_class(text, topic):
    return analysis.topic_word(text, topic)

def emotion_analysis_class(text):
    return analysis.emotion_analysis(text)


if __name__ == "__main__":

    print("Loading...")
    start = time.time()
    analysis_obj = analysis_all()
    end = time.time()
    print(f"It tooked : {end - start} seconds to load the model")
    print("model loaded!")
    # pred = analysis2_obj.topic_extraction("The U.S. Supreme Court’s decision Friday to end constitutional protection for abortion opened the gates for a wave of litigation.")
    
    sent = analysis_obj.sentiment_analysis("my hobby is drawing")
    question = analysis_obj.question_generation("my hobby is drawing","Drawing is a form of visual art in which an artist uses instruments to mark paper or other two-dimensional surface")
    pred = analysis_obj.topic_extraction("my hobby is drawing")
    wo = analysis_obj.topic_word("my hobby is drawing",pred)
    emo = analysis_obj.emotion_analysis("my hobby is drawing")
    

    # pr = ','.join(pred)
    print(sent)
    print(question)
    print(pred)
    print(wo)
    print(emo)
    