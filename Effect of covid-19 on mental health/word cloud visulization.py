# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 19:35:12 2020

@author: HP
"""

#load Dataset
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as pl

def load_data():
    data_=pd.read_csv('FINAL_MONTH_.csv')
    return data_

data__=load_data()
df_twitter= pd.DataFrame(data__)

def wordcloud_plot(wordcloud):
    pl.figure(figsize=(20,20))
    pl.imshow(wordcloud)
    pl.axis('off')
my_string=[]
for text in df_twitter['text']:
       my_string.append(text)
    
      
        
   
my_string=pd.Series(my_string).str.cat(sep=' ')    
wordcloud=WordCloud(width=1000,height=500).generate(my_string)
wordcloud_plot(wordcloud)