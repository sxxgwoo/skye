# python
import os
from uuid import uuid4
import inspect
# 3rd-party
import requests
import streamlit as st
from streamlit_lottie import st_lottie
from streamlit_option_menu import option_menu
from streamlit_chat import message

API_URL = "http://nlplab.iptime.org:51050/v1.0/chat"
headers = {"X-Auth": 'becec424c4eeb8510ad7c819b03889e671c01e87'}

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

#make it look nice from the start
st.set_page_config(initial_sidebar_state='collapsed',)

st.title('Semantic Retrieve Chatbot Demo')

lottie = load_lottieurl(
    'https://assets10.lottiefiles.com/packages/lf20_fnitdsu4.json')

st_lottie(lottie, height=200, key="hello")

st.write("Semantic Retrieve Chatbot Demo")

persona = option_menu(None, ["SKYE_v1", "caring", "enthusiastic", "friendly", "professional", "witty"], 
    icons=['robot', 'heart', "battery-charging", 'piggy-bank', 'building', 'emoji-sunglasses'], 
    menu_icon="cast", default_index=0, orientation="horizontal")

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

if 'history' not in st.session_state:
    st.session_state['history'] = []

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()

def get_text():
    input_text = st.text_input("You: ","Hello, how are you?", key="input")
    return input_text 


user_input = get_text()

if user_input:
    payload = {
        "persona": persona,
        "query": user_input,
        "k": 1,
        "score_threshold": 0.5,
        "retrieve_threshold": 0.8,
        "history": st.session_state.history[-4:]
    }
    output = query(payload)

    st.session_state.past.append(user_input)
    st.session_state.generated.append('{0}: {1}'.format(persona, output["bot_response"]))
    st.session_state.history.append(user_input)
    st.session_state.history.append(output["bot_response"])

if st.session_state['generated']:

    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')

st.write("PAYLOAD")
st.write(payload)
st.write("API RESPONSE")
st.write(output)


footer = """<style>
footer {visibility: hidden;}
.footer a{
color: rgb(46, 154, 255);
text-decoration: underline;
}
.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: black;
color: white;
z-index: 1;
text-align: center;
}
</style>
<div class="footer">
<p>Developed with ❤ by <a href="https://taeminlee.github.io/markdown-cv/" target="_blank">Taemin Lee</a></p>
</div>
"""
st.markdown(footer, unsafe_allow_html=True)