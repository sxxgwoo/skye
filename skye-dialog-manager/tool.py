from datetime import datetime
import pytz 
from bs4 import BeautifulSoup
import requests

def paraphrasing_question(question):
    question_temp = question.split()

    if 'I' in question_temp:
        index = question_temp.index('I')
        question_temp[index] = 'You'
    if 'my' in question_temp:
        index = question_temp.index('my')
        question_temp[index] = 'your'
    if 'i' in question_temp:
        index = question_temp.index('i')
        question_temp[index] = 'you'
    if 'me' in question_temp:
        index = question_temp.index('me')
        question_temp[index] = 'you'
    
    question = " ".join(question_temp)
    return question

def present_time(time_zone,type,region):
    time_format = '%Y-%m-%d %H:%M:%S'

    arrival_nyc = str(datetime.today())[:-7]
    nyc_dt_naive = datetime.strptime(arrival_nyc, time_format)
    eastern = pytz.timezone('Asia/Seoul')
    nyc_dt = eastern.localize(nyc_dt_naive)
    utc_dt = pytz.utc.normalize(nyc_dt.astimezone(pytz.utc)) #협정 세계시(utc로 normalize)

    region_tz = pytz.timezone(time_zone)
    region_time = region_tz.normalize(utc_dt.astimezone(region_tz))

    if type == 'time':
        return 'The local time in ' + region + ' is ' + region_time.strftime('%A, %r')
    else:
        return "Today's date in " + region + " is " + region_time.strftime('%A, %B %d')

def present_weather(region: str='Seoul'):
    try:
        url = "https://www.google.com/search?q=weather" + region +"&hl=en"
        html = requests.get(url).content
        soup = BeautifulSoup(html, 'html.parser')
        temp = soup.find('div', attrs={'class': 'BNeawe iBp4i AP7Wnd'}).text
        sky = soup.find('div', attrs={'class': 'BNeawe tAd8D AP7Wnd'}).text.split('\n')[1]
        answer =  'The weather in ' + region + ' is ' + sky + ' with a temperature of ' + temp

    except:
        answer = "Sorry, I don't know that region"

    return answer

