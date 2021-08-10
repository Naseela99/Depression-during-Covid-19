# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 10:50:48 2020

@author: HP
"""

""" Now to clean any dataset we are required to take in a few steps
1. load the dataset--done
2. remove the punctuation--done
3. remove stopwords--done
4. remove retweets--done
5. tokenize--done
6. stemming--done
7. lemmatization--done
8. Remove unnecessary tweets
~~~ TRY TO SORT OUT WITH DEPRESSION WORDSET~~~"""

#importing the necessary libraries
import pandas as pd
from nltk.stem.porter import *
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem.porter import *
import nltk
import string
import re
#nltk.download('wordnet')

pd.set_option('display.max_colwidth',100)


# Loading the dataset
def load_data_csv():
    #data_=pd.read_csv('predicted_post_covid.csv',names=['created_at','id','text'])
    #data_  = pd.read_csv('month.csv')
    data_  = pd.read_csv('tweet_data.csv')
    return data_

twitter_df=load_data_csv()
our_df=pd.DataFrame(twitter_df)
print(our_df.head())

# only include english tweets
our_df =our_df[our_df['lang']=='en']

#drop all retweets
our_df=our_df[~our_df['text'].str.startswith('RT')]
our_df['text']=our_df['text'].astype(str).str.replace('\d+','')
our_df=our_df[~our_df['text'].str.startswith('@')]
our_df['text']=our_df['text'].astype(str).str.replace('\d+','')


#remove the stop words... add few more
extended=['yr','year','woman','man','girl','boy','xe','x','xa','xm','b','still','let','know','case','new''due','time','lot','say','make','go','take','im','due','via','find','way','may','full','X',]
stpwrd=set().union(stopwords.words('english'),extended)
our_df['text']=our_df['text'].apply(lambda x: ' '.join(word for word in x.split() if word not in (stpwrd)))

#removing the hyperlinks
def remove_hyperlinks_retweets(txt):
    txt=' '.join([wrd for wrd in txt.split(' ') if 'http' not in wrd])
    return txt
our_df['text']=our_df['text'].apply(lambda x:remove_hyperlinks_retweets(x))
    
def remove_twitter_handles(txt):
    txt=' '.join([wrd for wrd in txt.split(' ') if '@' not in wrd])
    return txt
our_df['text']=our_df['text'].apply(lambda x:remove_twitter_handles(x))

#removing the punctuations
our_df['text']=our_df['text'].str.replace('[^\w\s]','')
our_df['text']=our_df['text'].str.replace("[^a-zA-Z#]",' ')
#Drop Colums Keep Only Tweets
our_df=our_df[['id','text']]


#stem_tweet=PorterStemmer()
#our_df['text']=our_df['text'].apply(lambda x:" ".join(stem_tweet.stem(txt) for txt in x.split()))
#our_df['text']=our_df['text'].astype(str).str.replace('\d+','')

#Lemmatization
ltizer=WordNetLemmatizer()
our_df['text']=our_df['text'].apply(lambda x:" ".join(ltizer.lemmatize(txt)for txt in x.split()))
our_df['text']=our_df['text'].astype(str)
our_df['text']=our_df['text'].apply(lambda y: " ".join([word for word in y.split() if word not in stopwords.words('english')]))


#save to a new file]
our_df.to_csv('tweet_data_cleaned.csv')
