import json
import os
class retrieve_senti:
    turns = 0


    def __init__(self,name,senti,turns = None):
        self.a = {}
        self.name = name
        self.senti = senti
        self.a['extended_answer'] = []
        self.a[1]=self.senti

        if os.path.exists('/home/sxxgwoo/skye/skye-dialog-manager/data/userinfo/data_'+name+'.json'):
            with open('/home/sxxgwoo/skye/skye-dialog-manager/data/userinfo/data_'+name+'.json','r', encoding='utf-8-sig') as json_file:
                json_data = json.load(json_file)
            tur = len(json_data)

            json_data[tur]=self.senti
            with open('/home/sxxgwoo/skye/skye-dialog-manager/data/userinfo/data_'+name+'.json','w', encoding='utf-8-sig') as f:
                json.dump(json_data,f,ensure_ascii=False,indent=4)
        else:
            with open('/home/sxxgwoo/skye/skye-dialog-manager/data/userinfo/data_'+name+'.json','w', encoding='utf-8-sig') as f:
                json.dump(self.a,f,ensure_ascii=False,indent=4)



def save_extended(name, extended_answer):
    if os.path.exists('/home/sxxgwoo/skye/skye-dialog-manager/data/userinfo/data_'+name+'.json'):
        with open('/home/sxxgwoo/skye/skye-dialog-manager/data/userinfo/data_'+name+'.json','r', encoding='utf-8-sig') as json_file:
            json_data = json.load(json_file)
            
        tur = len(json_data)

        json_data[str(tur-1)][0]['skye'] = json_data[str(tur-1)][0]['skye'] + " " + extended_answer
        if extended_answer == "":
            result = False
        elif extended_answer in json_data['extended_answer']:
            result = True
        else:
            json_data['extended_answer'].append(extended_answer)
            result = False

        with open('/home/sxxgwoo/skye/skye-dialog-manager/data/userinfo/data_'+name+'.json','w', encoding='utf-8-sig') as f:
            json.dump(json_data,f,ensure_ascii=False,indent=4)

        
        return result

    

    


