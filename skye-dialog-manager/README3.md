# postman TEST 사용시

### 1. url post 시 정확한 url 사용

- server가 열렸을때 나오는 url/v1.0/chat 
    ex) http://0.0.0.0:51050/v1.0/chat

- wsl 사용시 window ip주소가 아닌 wsl 할당받은 ip주소 사용
    (ifconfig, ipconfig 를 통해 주소값 확인가능)

### 2. Header에 TOKEN 넣기

- secret.py에 첫번째 api token 사용

- postman에 Headers tab 아래 KEY -> X-Auth, VALUE -> TOKEN입력

### 3. body 입력 형식

- /skye/skye-dialog-manager/openapi/dmlab.yaml
- /v1.0/chat 아래 requestBody의 properties 내용 확인
- example)
{
    "query": "do you remember what is my favorite food?",
    "user_name": "user_002",
    "retrieve_threshold": 0.8,
    "history":[""],
    "persona":"SKYE_v1",
    "k": 1,
    "score_threshold": 0.5   
}
- postman에 body tab에 -> raw -> json 형식으로 변경
- 위의 body 를 입력을 넣어주고 test 실행