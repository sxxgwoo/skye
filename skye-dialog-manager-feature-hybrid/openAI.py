import openai
import os
import json

def prompt_redesign(text_temp,name,term):
    if os.path.exists('/home/dmlab/skye/skye-dialog-manager/data/userinfo/data_'+name+'.json'):
        with open('/home/dmlab/skye/skye-dialog-manager/data/userinfo/data_'+name+'.json','r',encoding='utf-8-sig') as json_file:
            json_data = json.load(json_file)
        tr = len(json_data)-1
        i=1
        text_history=[]
        text_history.clear()
        text_history = str()
        while True:
            
            if term==-1:
                if i<=tr:
                    text_history += 'Human:'+json_data[str(i)][0]['user']+'\nAI:'+json_data[str(i)][0]['skye']+'\n'
                    text = text_history+'Human:'+text_temp+'\nAI:'
                    i+=1
                else:
                    break
            elif tr >= term>0:
                if i<=term:
                    text_history += 'Human:'+json_data[str(tr-term+i)][0]['user']+'\nAI:'+json_data[str(tr-term+i)][0]['skye']+'\n'
                    text = text_history+'Human:'+text_temp+'\nAI:'
                    i+=1
                else:
                    break            
            else:
                text ='\nHuman:'+text_temp+'\nAI:'
                break
        return text
    else:
        text ='\nHuman:'+text_temp+'\nAI:'
        
        return text

class skye_openAI:

    def __init__(self) -> None:

        openai.api_key = "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"

    def send(self, text: str, name: str, pre: str):
        self.pre = pre
        self.text = text
        self.name = name
        text_create = prompt_redesign(self.text, self.name, term=0)
        pre_condition = "The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly."+ self.pre+ "\n\n" + text_create

        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=pre_condition,
            temperature=0.9,
            max_tokens=150,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.6,
            stop=["Human:"]
        )

        res = response['choices'][0]['text'].replace("\n","")

        return res


def init():
    global openai_chatbot
    openai_chatbot = skye_openAI()
    
def run(**kwargs):
    # global openai_chatbot
    # openai_chatbot = skye_openAI()
    
    return openai_chatbot.send(**kwargs['body'])
    



