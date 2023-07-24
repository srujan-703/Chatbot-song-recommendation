import nltk
nltk.download()
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
from textblob import TextBlob
import requests
from keras.models import load_model
model = load_model('model.h5')
import json
import random
import text2emotion as te
intents = json.loads(open('data.json').read())
words = pickle.load(open('texts.pkl','rb'))
classes = pickle.load(open('labels.pkl','rb'))
def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res

# def get_track():
#     url1=f"http://ws.audioscrobbler.com/2.0/?method=tag.gettoptracks&tag=happy&api_key={'4a7050b3cd3983dfd80ef66a29caea7d'}&format=json&limit=10"
#     response1 = requests.get(url1)
#     payload1 = response1.json()
#     get_track=payload1['tracks']['track']
#     return render_template('songs.html', **get_track)

def get_emotion(ans):
    x= te.get_emotion(ans)
    global Keymax
    Keymax = max(zip(x.values(), x.keys()))[1]
    print(Keymax)

from flask import Flask, render_template, request

app = Flask(__name__)
app.static_folder = 'static'

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    get_emotion(userText)
    return chatbot_response(userText)

@app.route("/get_track")
def get_track():
    emotion=Keymax
    url1=f"http://ws.audioscrobbler.com/2.0/?method=tag.gettoptracks&tag={emotion}&api_key={'4a7050b3cd3983dfd80ef66a29caea7d'}&format=json&limit=10"
    response1 = requests.get(url1)
    payload1 = response1.json()
    get_track=payload1['tracks']['track']
    return render_template('songs.html',get_track=get_track)


if __name__ == "__main__":
    app.run()
    
    
    #0 import libraries
