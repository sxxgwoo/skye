# HT.V0.3.27

semantic retrieval based chitchat (디엠랩)


### 1. 필요 라이브러리 설치

- conda 설치

- `conda env create -f dmlab.yaml`

- `conda activate dmlab`

### 2. 데이터 준비

- [microsoft qnamaker chit-chat dataset](https://github.com/microsoft/botframework-cli/blob/main/packages/qnamaker/docs/chit-chat-dataset.md)을 다운로드 받고, paraphrasing하여 데이터 집합을 확장 합니다.

`cd data`

`bash get_chitchat.sh`

`python ans_para.py`

`cd ..`

### 3. 인코딩 및 인덱싱

- [SimCSE](https://github.com/princeton-nlp/SimCSE) 인코딩 후 [faiss](https://faiss.ai/)로 인덱싱하여 저장합니다.

- 인코딩 및 인덱싱 파일 목록은 `/configs/data/encode.yaml`의 정의를 따릅니다.

`python encode.py`

### 4. API 서버 실행

- `configs/server/default.yaml` 에 정의된 ip와 port로 API 서버를 실행합니다.

- 사용 데이터베이스는 `/configs/data/serve.yaml`의 정의를 따릅니다.

- API 서버 swagger URL : `http://[IP]:[PORT]/v1.0/ui`

`python app.py`

- API 서버에서 사용하는 key는 `secret.py` 에서 정의합니다.

### 5. 데모 웹서비스 실행

`streamlit run demo.py`