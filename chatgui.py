import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
from keras.models import load_model
import json
import random

lemmatizer = WordNetLemmatizer()

model = load_model('chatbot_model.h5')

intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))


def arrange_sentence(sentence):

    sentence_words = nltk.word_tokenize(sentence)

    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = arrange_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1

    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR]
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

print("Chatbot is Running")

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res


#Creating GUI with tkinter
import tkinter
from tkinter import *


def send():
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)

    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12 ))
        res = chatbot_response(msg)
        ChatLog.insert(END, "Buddy: " + res + '\n\n')
        ChatLog.yview(END)
 

body = Tk()
body.title("BUDDY")
body.geometry("380x552")

'''
img = PhotoImage(file="bot.jpg")
photo_L= Label (body, image=img)
photo_L.pack(pady=5)
'''
#Create Chat window
ChatLog = Text(body, bd=0, bg="pink", height="5", width="50", font="Arial",)


#Create Button to send message
SendButton = Button(body, font=("Verdana",8,'bold'), text="Send", width="5", height="1",
                    bd=0, bg="#800040", activebackground="#330033",fg='#ffffff',
                    command= send )

#Create the box to enter message
EntryBox = Text(body, bd=0, bg="white",width="29", height="5", font="Arial")

#Place all components on the screen
ChatLog.place(x=6,y=6, height=590, width=370)
EntryBox.place(x=95, y=452, height=80, width=265)
SendButton.place(x=30, y=469, height=45)

body.mainloop()
