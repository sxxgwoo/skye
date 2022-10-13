import os
import glob
import json
from textwrap import indent
import tqdm
import pandas as pd
import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

print('initialize paraphrasing model')

overwrite = False

# pre-training 된 pegasus transformer모델 사용
model_name = 'tuner007/pegasus_paraphrase'
torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(torch_device)

# 1개의 문장을 paraphrasing하여 5개의 문장을 만드는 함수
# 인자로 return 하는 sentence수와 디코딩 전략인 beam search에서 가능성있는 sequence의 개수
# beam search = 5: 가장 높은 확률을 가지는 5개의 sequence를 항상 유지한다. 
def get_response(input_text,num_return_sequences,num_beams):
  batch = tokenizer([input_text],truncation=True,padding='longest',max_length=60, return_tensors="pt").to(torch_device)
  translated = model.generate(**batch,max_length=60,num_beams=num_beams, num_return_sequences=num_return_sequences, temperature=1.5)
  tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
  return tgt_text

# paraphrasing한 문장을 return하는 함수
def get_para(sentence):
    para = get_response(sentence, num_return_sequences=5, num_beams=5)
    if sentence not in para:
        para.append(sentence)
    return para

# file을 불러오는 함수
def get_answer_groups(filename):
    if filename.endswith('.tsv'):
        db = pd.read_csv(filename, delimiter='\t')
    elif filename.endswith('.csv'):
        db = pd.read_csv(filename)
    answer_groups = db.groupby(db['Answer'])      # Question을 Answer별로 grouping
    # print(len(answer_groups)) # 90 answers
    return answer_groups

for file in glob.glob('*.csv') + glob.glob('*.tsv'):    # glob 모듈은 인자로 받은 패턴과 이름이 일치하는 모든 파일과 디렉터리의 리스트를 반환
    print(file)
    answers_filename = 'answers_' + file.replace(file[-4:], '.json')
    extended_filename = 'extended_' + file.replace(file[-4:], '.json')    # paraphrasing 결과를 저장할 json파일 이름을 만듦              
    if not overwrite and os.path.exists(answers_filename):
        print('already processed!')                
        continue                                  # 이미 paraphrasing 처리된 문장이 존재하면 for문의 처음으로
    extended_answers = {}
    extended_FAQ = {'version': '1.0.0', 'data':[]}
    answer_groups = get_answer_groups(file)        # 이 함수를 통해 db에서 aanswer별로 구분한(grouping한) data를 받음
    answer_id = 0                                                           # tqdm 모듈은 진행되는 상황을 시각적으로 알려줌 
    for answer, answer_group in tqdm.tqdm(answer_groups):                   # answer는 그룹 label(answer), answer_group = (Answer, Question)
        extended_answers[answer] = get_para(answer)                 # answer를 paraphrasing하여 dictionary형태로 저장(key: answer, value: para_answer)
        for question_id, question_orig in enumerate(answer_group['Question']):
            for question_para_id, question_para in enumerate(get_para(question_orig)):
                extended_FAQ['data'].append({
                    'id': '{0}_{1}_{2}'.format(answer_id, question_id, question_para_id),       # question을 paraphrasing 하여 original answer, question과 함께
                    'question': question_para,                                                  # paraphrasing된 qeustion을 저장
                    'answer': answer
                })
        answer_id += 1
    with open(answers_filename, 'w') as fp:
        json.dump(extended_answers, fp, indent=4)
    with open(extended_filename, 'w') as fp:
        json.dump(extended_FAQ, fp, indent=4)

