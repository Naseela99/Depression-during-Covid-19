
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import *
from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer


def load_csv_file():
    file_csv =  pd.read_csv('month_clean_1.csv')
    return file_csv

data_tweet=load_csv_file()
data_f= pd.DataFrame(data_tweet)

#removal of stop words
def removal_of_stopwords(txt):
    txt=[wrd for wrd in txt.split() if wrd not in stopwords.words('english')]
    return txt
#data_f['text']=data_f['text'].apply(lambda y:removal_of_stopwords(y))
#data_f['text']=data_f['text'].astype(str).str.replace('\d+','')
    

#Stemming
stem_tweet=PorterStemmer()
data_f['text']=data_f['text'].apply(lambda x:" ".join(stem_tweet.stem(txt) for txt in x.split()))
#our_df['text']=our_df['text'].astype(str).str.replace('\d+','')


#Break into tokens 
twk=TweetTokenizer(strip_handles=True,preserve_case=False,reduce_len=True)
data_f['text']=data_f['text'].apply(twk.tokenize)
data_f['text']=data_f['text'].astype(str).str.replace('\d+','')  

#removing the punctuations
data_f['text']=data_f['text'].str.replace('[^\w\s]','')
data_f['text']=data_f['text'].str.replace("[^a-zA-Z#]",' ')

#Lemmatization
ltizer=WordNetLemmatizer()
data_f['text']=data_f['text'].apply(lambda x:" ".join(ltizer.lemmatize(txt)for txt in x.split()))
data_f['text']=data_f['text'].astype(str)
data_f['text']=data_f['text'].apply(lambda y: " ".join([word for word in y.split() if word not in stopwords.words('english')]))

data_f.to_csv('FINAL_DATA_CLEANED_MONTH.csv')