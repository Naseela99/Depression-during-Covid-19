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

from nltk.corpus import stopwords
from nltk.stem.porter import *

import nltk
import string
import re
#nltk.download('wordnet')

pd.set_option('display.max_colwidth',100)


#Defining Keywords-Depressed
depressed_keywords=['antidepressants','hopeless','anxiety','bipolar','dysthymia','electroconclusive','hypomania','depress','depression','mania','insomnia','panic','psychotherapy','psychiatrist','fault','lonely','alone','sad','distress','gloom','unhappy','melancholia','misery','suicide','anhedonia','delusion','hallucination','menatl-health','mentalhealth']


#Defining keywords- Non-depressed
non_depressed_keywords=['hope','cheer','glad','peace','comfort','happy','blessing','satisfied','joy','bliss','delight','merry','euphoria','jubilant','optimistic','optimist','laugh']



#combined keywords
all_keywords= depressed_keywords+non_depressed_keywords
keyword_all='|'.join(all_keywords)


# Loading the dataset
def load_data_csv():
    #data_=pd.read_csv('day.csv')
    data_=pd.read_csv('month.csv')
    return data_

twitter_df=load_data_csv()
our_df=pd.DataFrame(twitter_df)


# only include english tweets
our_df =our_df[our_df['lang']=='en']

#drop all retweets
our_df=our_df[~our_df['text'].str.startswith('RT')]
our_df['text']=our_df['text'].astype(str).str.replace('\d+','')


#keep only keywords~~ depressed , non-depressed and neutral tweets
our_df=our_df[our_df.text.str.contains(keyword_all)]

#remove the stop words... add few more
extended=['yr','year','woman','man','girl','boy','still','let','know','case','new''due','time','lot','say','make','go']
stpwrd=set().union(stopwords.words('english'),extended)
our_df['text']=our_df['text'].apply(lambda x: ' '.join(word for word in x.split() if word not in (stpwrd)))


#removing the hyperlinks
def remove_hyperlinks_retweets(txt):
    txt=' '.join([wrd for wrd in txt.split(' ') if 'http' not in wrd])
    return txt
our_df['text']=our_df['text'].apply(lambda x:remove_hyperlinks_retweets(x))



#Drop Colums Keep Only Tweets
our_df=our_df[['id','text']]


#save to a new file]
#our_df.to_csv('day-clean-final.csv')
our_df.to_csv('month_clean.csv')
