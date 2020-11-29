from waitress import serve
from pyramid.config import Configurator
from pyramid.response import Response

class predictions():
	def predict_sentiment(text):
		import pandas as pd
		import numpy as np
		from nltk.tokenize import word_tokenize
		from nltk.stem import WordNetLemmatizer
		from nltk import pos_tag
		from nltk.corpus import stopwords
		from sklearn.preprocessing import LabelEncoder
		from collections import defaultdict
		from nltk.corpus import wordnet as wn
		from sklearn.feature_extraction.text import TfidfVectorizer
		from sklearn import model_selection, naive_bayes
		from sklearn.svm import SVC 
		from sklearn.metrics import accuracy_score
		import pickle
		import random
		import re
		import nltk.stem
		from nltk.stem import SnowballStemmer 
		from nltk import stem
		from nltk.stem import WordNetLemmatizer
		from nltk.tokenize.toktok import ToktokTokenizer
		from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
		from sklearn.metrics import roc_curve,auc
		from scipy import interp
		import matplotlib.pyplot as plt
		from itertools import cycle
		from sklearn.ensemble import RandomForestClassifier
		import seaborn as sns 
		from nltk.tokenize import word_tokenize  
	
		main_df = pd.read_csv('airline_sentiment_analysis.csv')
		main_df = main_df[['airline_sentiment','text']]
		label= main_df['airline_sentiment']
		label= list(label)
	
		for i in range(len(label)):
			if(label[i] =='positive'):
				label[i]= 0
			else:
				label[i]= 1
		del(main_df['airline_sentiment'])
		main_df['Sentiment']= label
		main_df['text'] = [entry.lower() for entry in main_df['text']]
		REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
		BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
	
		def Clean_It(text):
			text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
			text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
			return text
		main_df['text'] = main_df['text'].apply(Clean_It)
		
		contraction_dict = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would","youve": "you have", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have"}

		def _get_contractions(contraction_dict):
			contraction_re = re.compile('(%s)' % '|'.join(contraction_dict.keys()))
			return contraction_dict, contraction_re
	
		contractions, contractions_re = _get_contractions(contraction_dict)
	
		def replace_contractions(text):
			def replace(match):
				return contractions[match.group(0)]
			return contractions_re.sub(replace, text)
		main_df['text']= main_df['text'].apply(replace_contractions)
	
		def lemmatize_stemming(text):
			stemmer = SnowballStemmer("english")
			return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))
		main_df['text']= main_df['text'].apply(lemmatize_stemming)
		Normalized_Reviews=[]
		for i in range(len(main_df['text'])):
			stop_words = set(stopwords.words('english'))
			word_tokens = word_tokenize(main_df['text'][i])
			filtered_sentence = [w for w in word_tokens if not w in stop_words]  
			filtered_sentence = []
			for w in word_tokens:  
				if w not in stop_words:  
					filtered_sentence.append(w)
			Normalized_Reviews.append(filtered_sentence)
		Normalized_Reviews= pd.Series(Normalized_Reviews)
		main_df['text']= Normalized_Reviews
		main_df.to_csv('Normalized_Airlines.csv',index= False)
		main_df= pd.read_csv('Normalized_Airlines.csv')
		##Splitting the data into training and testing
		Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(main_df['text'],main_df['Sentiment'],test_size=0.3)
		
		Tfidf_vect = TfidfVectorizer(max_features=5000)
		Tfidf_vect.fit(main_df['text'])
		Train_X_Tfidf = Tfidf_vect.transform(Train_X)
		Test_X_Tfidf = Tfidf_vect.transform(Test_X)
		#print(Tfidf_vect.vocabulary_)
		
		from sklearn.neighbors import KNeighborsClassifier
		Knn = KNeighborsClassifier()
		Knn.fit(Train_X_Tfidf,Train_Y)
		prediction = Knn.predict(text)
		if prediction:
			return Response('<body><h1>Positive</h1></body>')
		else:
			return Response('<body><h1>Negative</h1></body>')
	
if __name__ == '__main__':
	with Configurator() as config:
		obj= predictions()
		config.add_route('classify','/classify/{text}')
		config.add_view(obj.predict_sentiment,route_name='classify')
		app = config.make_wsgi_app()
	serve(app, host='0.0.0.0', port=6543)
