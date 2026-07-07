# Minimal working English-French Many-to-Many RNN example
import os,re,pickle,numpy as np,pandas as pd,streamlit as st
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Embedding,SimpleRNN,TimeDistributed,Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
MODEL="translator_model.keras"
INPUT_TOKENIZER="input_tokenizer.pkl"
TARGET_TOKENIZER="target_tokenizer.pkl"
MAX_WORDS=5000
MAX_LEN=10

def clean_text(t):
    t=str(t).lower()
    t=re.sub(r"[^a-z0-9 ]"," ",t)
    return re.sub(r"\s+"," ",t).strip()

def train_model():
    df=pd.read_csv("en-fr.csv",encoding="utf-8-sig")
    df.columns=df.columns.str.strip().str.replace("ï»¿","",regex=False)
    df=df[["English","French"]]
    df["English"]=df["English"].apply(clean_text)
    df["French"]=df["French"].apply(clean_text)
    it=Tokenizer(num_words=MAX_WORDS,oov_token="<OOV>")
    ot=Tokenizer(num_words=MAX_WORDS,oov_token="<OOV>")
    it.fit_on_texts(df["English"])
    ot.fit_on_texts(df["French"])
    X=pad_sequences(it.texts_to_sequences(df["English"]),maxlen=MAX_LEN,padding="post")
    Y=pad_sequences(ot.texts_to_sequences(df["French"]),maxlen=MAX_LEN,padding="post").reshape(len(df),MAX_LEN,1)
    pickle.dump(it,open(INPUT_TOKENIZER,"wb"))
    pickle.dump(ot,open(TARGET_TOKENIZER,"wb"))
    xtr,xte,ytr,yte=train_test_split(X,Y,test_size=.2,random_state=42)
    m=Sequential([Embedding(MAX_WORDS,128,input_length=MAX_LEN),SimpleRNN(128,return_sequences=True),TimeDistributed(Dense(MAX_WORDS,activation="softmax"))])
    m.compile("adam","sparse_categorical_crossentropy",metrics=["accuracy"])
    m.fit(xtr,ytr,epochs=10,batch_size=16,validation_split=.2)
    m.save(MODEL)

def translate_sentence(s):
    m=load_model(MODEL)
    it=pickle.load(open(INPUT_TOKENIZER,"rb"))
    ot=pickle.load(open(TARGET_TOKENIZER,"rb"))
    seq=pad_sequences(it.texts_to_sequences([clean_text(s)]),maxlen=MAX_LEN,padding="post")
    ids=np.argmax(m.predict(seq,verbose=0),axis=-1)[0]
    rev={v:k for k,v in ot.word_index.items()}
    return " ".join(rev[i] for i in ids if i in rev)

if not os.path.exists(MODEL): train_model()
st.title("English to French Translator")
msg=st.text_input("Enter English")
if st.button("Translate"): 
    st.success(translate_sentence(msg))
