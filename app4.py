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
from streamlit_modal import Modal
import seaborn as sns
import warnings
from annotated_text import annotated_text
from annotated_text import annotated_text, annotation

warnings.filterwarnings("ignore")
stopwords_set = set(stopwords.words('english'))
st.set_option('deprecation.showPyplotGlobalUse', False)
text=""
text1=""
text2=""
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
#predict_button = st.button("Predict")
#st.write(predict_button)


	#st.write('Polarity: ', round(blob.sentiment.polarity,2))
	#st.write('Subjectivity: ', round(blob.sentiment.subjectivity,2))
	plt.figure(figsize = (20,20))

redunant_df = pd.read_csv("reduced_words.csv")
redunant=set(redunant_df["words"])

def annotating_text(text):
	annot_text=list()
	for word in str(text).split():
		if word in stopwords_set:
			annot_text.append((word, "Stop Word", "#fea","#010203"))
		else:
			annot_text.append(word)
	return annot_text

annot_text=annotating_text(text)
annotated_text(annot_text)

text1=cleaning_reduntant(text1)
if(text1!=""):
	#st.title("Cleaned Text")
	text1 = re.sub('((www.[^s]+)|(https?://[^s]+))|(http?://[^s]+)', '',text1)
	tknzr = TweetTokenizer(strip_handles=True)
	text1=tknzr.tokenize(text1)
	text1=str(text1)
	text1=re.sub(r'[^a-zA-Z0-9\s]', '', text1)
	text2=str(text1)
	text1=cleantext.clean(text1, clean_all= False, extra_spaces=True ,stopwords=True ,lowercase=True ,numbers=True , punct=True)
	#st.write(text1)

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
button=st.button("Stats For Nerd")	
modal = Modal("Stats For Nerd","black")

if button:
    modal.open()
import streamlit.components.v1 as components
if modal.is_open():
	
	with modal.container():
		#st.write("Prediction Model Used - Support Vector Classification (SVC)")
		#st.write("Word Embedding done Using - Pre-Trained Glove")
		#button1=st.button("Classification Report")
		col1, col2,col3 = st.columns(3)
		if col1.button('Classification Report','Classification Report'):
			st.write("Classification Report of Trained Model")
			book1=pd.read_csv("Book1.csv")
			book1.replace(np.NaN,"",inplace=True)
			#book1 = book1.style.set_properties(**{'text-align': 'center'})
			html_table = book1.to_html(index=False)
			html_table = html_table.replace('<table', '<table style="border-collapse: collapse;"')
			st.markdown(html_table, unsafe_allow_html=True)
		if col2.button('Confusion Matrix','Confusion Matrix'):
			st.write("Confusion Matrix of Trained Model")
			categories = ['figurative', 'irony', 'regular', 'sarcasm']
			plt.figure(figsize=(7, 5))
			df_cm=pd.read_csv('Confusion Matrix.csv')
			df_cm.index=['figurative', 'irony', 'regular', 'sarcasm']
			sns.heatmap(df_cm, annot=True,cmap = 'Blues',fmt = '.1f',xticklabels = categories, yticklabels = categories)
			plt.xlabel("Predicted values", fontdict = {'size':14}, labelpad = 10)
			plt.ylabel("Actual values" , fontdict = {'size':14}, labelpad = 10)
			plt.title ("Confusion Matrix", fontdict = {'size':18}, pad = 20)
			st.pyplot()
		if col3.button('Word Cloud','Word Cloud'):
			if (text2!=""):
				wordcloud = WordCloud(width=800, height=400).generate(text)
				st.title('Word Cloud')
				plt.figure(figsize=(7, 5))
				plt.imshow(wordcloud, interpolation='bilinear')
				plt.axis('off')
				st.pyplot(plt)
			else:
				st.write("No Words to Plot")
		
		st.write("[Click Here to view complete GitHub Repository](https://github.com/VinayNagaraj07/Twitter-Sentiment-Analysis)")
		
st.markdown('''
    ## Disclaimer
    
    This Predictions are made from training on a specific Dataset only and for it is to be used solely learning purposes only. Please consult with a qualified professional before making any decisions.
    
    ---
    ''')
