# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 18:44:51 2021

@author: HP
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
def load_csv_file():
    #file_csv=pd.read_csv('dataset2.csv')
    file_csv=pd.read_csv('FINAL_DAY.csv')
    return file_csv

data_file=load_csv_file()
data_to_train=pd.DataFrame(data_file)

#data_file_test= pd.read_csv('month_clean_1.csv')
data_file_test= pd.read_csv('tweet_data_cleaned.csv')
data_to_label=pd.DataFrame(data_file_test)

extended=['many','one','well','keep','everyone','new','people','today','yr','year','woman','man','girl','boy','still','let','know','case','new''due','time','lot','say','make','go','covid','coronavirus','day','thing','viru','u']
stpwrd=set().union(stopwords.words('english'),extended)
data_to_train['text']=data_to_train['text'].apply(lambda x: ' '.join(word for word in x.split() if word not in (stpwrd)))
data_to_label['text']=data_to_label['text'].apply(lambda x: ' '.join(word for word in x.split() if word not in (stpwrd)))
#label encoder
encoder_label=LabelEncoder()
Y=encoder_label.fit_transform(data_to_train['label'].astype(str))

#VECTORIZATION TF-IDF
tfidfcon=TfidfVectorizer(ngram_range=(1,3))

X=data_to_train['text'].values

X = tfidfcon.fit_transform(data_to_train['text']).toarray()
data_to_train['vectorized']=list(X)

data_to_train.to_csv("FINAL_DAY_.csv")

X1=data_to_label['text'].values
X1=tfidfcon.fit_transform(data_to_label['text']).toarray()
data_to_label['vectorized']=list(X1)

#SVC classifier
classifier=SVC(kernel='linear')
for i in range(100):
    classifier.fit(X,Y)
prediction=classifier.predict(X1)
data_to_label['label']=prediction

data_to_label.to_csv('FINAL_.csv')
