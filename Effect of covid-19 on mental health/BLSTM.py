#import all the necessary libraries
import keras
from sklearn.preprocessing import LabelEncoder
from keras.layers import Dropout, Dense, Embedding, LSTM, Bidirectional
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras import Sequential
from sklearn.metrics import matthews_corrcoef, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.utils import shuffle
import numpy as np
import pickle
import matplotlib.pyplot as plt
from nltk.tokenize import TweetTokenizer
import warnings
import logging
import pandas as pd
logging.basicConfig(level=logging.INFO)

#import dataframe and drop unnecessary columns

mydata=pd.read_csv('FINAL_MONTH_.csv')
mydataframe=pd.DataFrame(mydata)
del mydataframe['Unnamed: 0']
del mydataframe['Unnamed: 0.1']
del mydataframe['Unnamed: 0.1.1']
#print(mydataframe.head()) 


#label encoder on the target variable
encoder=LabelEncoder()
mydataframe['label']=encoder.fit_transform(mydataframe['label'])

# the preprocessing function to prepare our text for training and testing
def model_preprocessing(train_data,test_data,MAX_WORDS=80000, MAX_SEQ_LEN=1000):
    np.random.seed(7)
    txt=np.concatenate((train_data,test_data),axis=0)
    txt=np.array(txt)
    tok=Tokenizer(num_words=MAX_WORDS)
    tok.fit_on_texts(txt)
    pickle.dump(tok,open('text_tokenizer.pkl','wb'))
    seq=tok.texts_to_sequences(txt)
    word_index=tok.word_index
    txt=pad_sequences(seq,maxlen=MAX_SEQ_LEN)
    ind=np.arange(txt.shape[0])
    ##np.random.shuffle(ind)
    txt=txt[ind]
    train_data_Glove=txt[0:len(train_data), ]
    test_data_Glove=txt[len(train_data):, ]
    dictionary_embedding={}
    func=open("glove.6B.50d.txt",encoding="utf8")
    for l in func:
        val=l.split()
        wrd=val[0]
        try:
            coefficients=np.asarray(val[1:], dtype='float32')
        except:
            pass
        dictionary_embedding[wrd]=coefficients
    func.close()
    return(train_data_Glove,test_data_Glove,word_index,dictionary_embedding)

def model_building_blstm(word_index, dictionary_embedding,nclasses, MAX_SEQ_LEN=1000, dropout=0.2, hidden_layer=3, lstm_node=32,EMBEDDING_DIMS=50):
    blstm_model=Sequential() # model is sequntial
    matrix_embedding=np.random.random((len(word_index)+1,EMBEDDING_DIMS)) # make embedding matrix
    for wrd,i in word_index.items():
        #print(wrd)
        #print("\n")
        vector_embedding=dictionary_embedding.get(wrd)
        if vector_embedding is not None:
            if len(matrix_embedding[i]) != len(vector_embedding):
                print("errorrrrr")
                exit(1)
            matrix_embedding[i]=vector_embedding
    
    #Now add an embedding layer
    blstm_model.add(Embedding(len(word_index)+1,EMBEDDING_DIMS,weights=[matrix_embedding],input_length=MAX_SEQ_LEN,trainable=True))
    
    #Hidden layers - for every layer bi-lstm followed by dropout layer
    for i in range (0,hidden_layer):
        blstm_model.add(Bidirectional(LSTM(lstm_node,recurrent_dropout=0.2,return_sequences=True)))
        blstm_model.add(Dropout(dropout))
    blstm_model.add(Bidirectional(LSTM(lstm_node,recurrent_dropout=0.2,return_sequences=False)))
    blstm_model.add(Dropout(dropout))
    blstm_model.add(Dense(256,activation='relu')) #fully connected layer
    blstm_model.add(Dense(nclasses,activation='softmax'))
    blstm_model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return blstm_model

X=mydataframe['text']
Y=mydataframe['label']
train_data, test_data, train_data_y, test_data_y=train_test_split(X,Y,test_size=0.2)
train_data_Glove, test_data_Glove, word_index, dictionary_embedding= model_preprocessing(train_data, test_data) 
blstm_model= model_building_blstm(word_index, dictionary_embedding,3)
blstm_model.summary()

#blstm_model=pickle.load(open('blstm_model.pkl','rb'))
# TRAINING AND EVALUATING
def evaluating_report(labels,predictions):
    matt_coeff=matthews_corrcoef(labels,predictions)
   
    confusion_=confusion_matrix(labels,predictions)
    
    fp=confusion_.sum(axis=0)-np.diag(confusion_)
    fn=confusion_.sum(axis=1)-np.diag(confusion_)
    tp=np.diag(confusion_)
    tn=confusion_.sum()-(fp+fn+tp)
    
    
        
    precision=tp/(tp+fn)
    recall=tp/(tp+fn)
    f1=(2*(precision*recall))/(precision+ recall)
    return{
        "matthews_correlation_coefficient" : matt_coeff,
        "true positive": tp,
        "true negative":tn,
        "false positive":fp,
        "false negative":fn,
        "precision": precision,
        "recall": recall,
        "F1": f1,
        "accuracy": (tp+tn)/(tp+tn+fp+fn)
        
        
        }

def computing_metrics(labels,predictions):
    assert len(predictions)==len(labels)
    return evaluating_report(labels,predictions)

def graph_plotting(history,string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' +string],'')
    plt.xlabel('Epochs')
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()
    
history=blstm_model.fit(train_data_Glove,train_data_y,validation_data=(test_data_Glove,test_data_y),epochs=5,batch_size=64,verbose=1)
graph_plotting(history, 'accuracy')
graph_plotting(history, 'loss')

pickle.dump(blstm_model,open('blstm_model.pkl','wb'))
    
predicted=blstm_model.predict_classes(test_data_Glove)
print(metrics.classification_report(test_data_y, predicted))
print('\n')
logger=logging.getLogger('logger')
results=computing_metrics(test_data_y, predicted)
for k in (results.keys()):
    logger.info('%s =%s',k,str(results[k]))
     
    
        
    

    
    