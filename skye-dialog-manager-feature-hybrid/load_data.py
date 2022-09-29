import json

def load_sentiment(user_name):
    with open('/home/dmlab/skye/skye-dialog-manager/data/userinfo/data_'+user_name+'.json', encoding='utf-8-sig') as f:

        json_data = json.load(f)
        
    return json_data


def load_answer():
    with open('/home/dmlab/skye/skye-dialog-manager/data/data_qg_answer.json','r', encoding='utf-8-sig') as f:

        json_data = json.load(f)
    
    return json_data