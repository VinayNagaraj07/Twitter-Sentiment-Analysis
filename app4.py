from textblob import TextBlob
import pandas as pd
import streamlit as st
import cleantext
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import time
import re
from nltk.tokenize import TweetTokenizer
import matplotlib.pyplot as plt
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=200)

stopwords_set = set(stopwords.words('english'))

text=""
text1=""
filename="sentiment_model2.pkl"
filename1="vectorizer_model.pkl"

with open(filename1, 'rb') as file:
    vectorizer = pickle.load(file)

def display_sarcastic_remark(remark):
    st.title(remark)
    time.sleep(0.1)
def cleaning_reduntant(text):
	return " ".join([word for word in str(text).split() if word not in redunant])

st.header('Sentiment Analysis')
with st.title('Analyze Text'):
	text = st.text_input('Text here: ')
if text:
	text1=text
	blob = TextBlob(text)
predict_button = st.button("Predict")
st.write(predict_button)
if (predict_button==True):
	if (text!=""):
		wordcloud = WordCloud(width=800, height=400).generate(text)
		st.title('Word Cloud')
		plt.figure(figsize=(10, 5))
		plt.imshow(wordcloud, interpolation='bilinear')
		plt.axis('off')
		st.pyplot(plt)

		#st.write('Polarity: ', round(blob.sentiment.polarity,2))
		#st.write('Subjectivity: ', round(blob.sentiment.subjectivity,2))
		plt.figure(figsize = (20,20))

	redunant_df = pd.read_csv("reduced_words.csv")
	redunant=set(redunant_df["words"])

	text1=cleaning_reduntant(text1)
	if(text1!=""):
		st.title("Cleaned Text")
		text1 = re.sub('((www.[^s]+)|(https?://[^s]+))|(http?://[^s]+)', '',text1)
		tknzr = TweetTokenizer(strip_handles=True)
		text1=tknzr.tokenize(text1)
		text1=str(text1)
		text1=re.sub(r'[^a-zA-Z0-9\s]', '', text1)
		text1=cleantext.clean(text1, clean_all= False, extra_spaces=True ,stopwords=True ,lowercase=True ,numbers=True , punct=True)
		st.write(text1)

	with open(filename, 'rb') as file:
		model = pickle.load(file)
		unseen_tweets=[text1]
		unseen_df=pd.DataFrame(unseen_tweets)
		unseen_df.columns=["Unseen"]

	X_test = vectorizer.transform(unseen_tweets)
	y_pred = model.predict(X_test)

	if text!="":
		if(y_pred==0):
			remark = "That's Figurative!😄"
			display_sarcastic_remark(remark)
		if(y_pred==1):
			remark = "That's Irony!😏"
			display_sarcastic_remark(remark)
		if(y_pred==2):
			remark = "That's Regular!😐"
			display_sarcastic_remark(remark)
		if(y_pred==3):
			remark = "That's Sarcasm!🙃"
			display_sarcastic_remark(remark)
	else:
		st.write(text1)
		remark = "No Words to Analyze"
		display_sarcastic_remark(remark)
