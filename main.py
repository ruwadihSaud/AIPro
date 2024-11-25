import streamlit as st
import pickle
from sklearn.compose import make_column_transformer
import pandas as pd
import datetime
from PIL import Image, ImageDraw
import time
import base64
import sklearn
from sklearn.ensemble import RandomForestClassifier
import nltk
import sklearn
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from streamlit_option_menu import option_menu
from streamlit_lottie import st_lottie
import requests
import json
from urllib.request import urlopen
import time
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import arabic_reshaper
from ar_wordcloud import ArabicWordCloud
from bidi.algorithm import get_display
# Modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Embedding, Dropout, GlobalAveragePooling1D, Flatten, SpatialDropout1D, Bidirectional, GRU, Dense, Conv1D, MaxPooling1D, GlobalMaxPooling1D # Importing GlobalMaxPooling1D
from sklearn.base import BaseEstimator, TransformerMixin
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.compose import make_column_transformer
import numpy as np



st.set_page_config(page_title="Noty Deduction",page_icon="logo.png", initial_sidebar_state="auto", menu_items=None,layout="centered")


# to set background 
@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img_background = get_img_as_base64("back.png")


# css section
page_bg_img = f"""
<style>

[data-testid="stAppViewContainer"] > .main {{
background-size: 100%;
background-repeat: repeat;
background-position: center;
background-attachment: local;  /*fixed*/
background-image: url("data:image/png;base64,{img_background}");
}}

</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

#open css style file
with open('style.css') as f:
    css = f.read()
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)

#Text Preprocessor class
class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, max_words=1000, max_len=10):
        self.max_words = max_words
        self.max_len = max_len
        self.tokenizer = Tokenizer(num_words=500, char_level = False, oov_token = '<OOV>')
    
    def fit(self, X, y=None):
        # Fit the tokenizer on the text data
        self.tokenizer.fit_on_texts(X.squeeze())
        return self
    
    def transform(self, X, y=None):
        # Transform text to sequences and pad them
        X_seq = self.tokenizer.texts_to_sequences(X.squeeze())
        X_pad = pad_sequences(X_seq, maxlen=self.max_len)
        return X_pad
#Define Column Transformer
preprocessor = make_column_transformer(
    (TextPreprocessor(max_words=1000, max_len=50), 'tweet')  # Process 'text_data' column
)

#load model and transformer data
model = pickle.load(open("my_model","rb"))
lstm = pickle.load(open("my_LSTM","rb"))
#transformer = pickle.load(open('transformer', 'rb'))

#the page

#the time header
header = st.container()
timestamp = datetime.datetime.now()
if timestamp.hour<12:timestamp = "Good Morning"
elif timestamp.hour < 16:timestamp = "Good afternoon"
else:timestamp = "Good evening"

header.write("<div class='fixed-header'>{} </div>".format(timestamp), unsafe_allow_html=True)
st.markdown('#')
st.markdown('#')

#the menu

selected = option_menu(None, ["Home","Uploaded File", 'About' , 'Creators'],icons=['house', "cloud-upload", 'info-square','people'], menu_icon="cast", default_index=0, orientation="horizontal",styles={"nav-link": {"--hover-color": "#c4324120"}})


#'''Home page''' 
if selected == "Home":
    
    st.markdown('#')
    
    #the title of the page
    st.markdown('<h1 class="title">Offensive Language Detector</h1>', unsafe_allow_html=True)
    st.markdown('#')
    
    dtext = st.text_area("",label_visibility="collapsed",placeholder="Enter the text")

    st.markdown('#')
    col1, col2, col3 , col4, col5 = st.columns(5)
    
    with col3:
        button = st.button("**prediction**")
        if button:
            st.markdown('#')
            
            tokenizer = Tokenizer(num_words = 500, char_level = False, oov_token = '<OOV>')
            tokenizer.fit_on_texts([dtext])
            training_sequences = tokenizer.texts_to_sequences([dtext])
            training_padded = pad_sequences(training_sequences,maxlen = 50, padding = 'post', truncating = 'post')
             
            y_pred = model.predict(training_padded)
            prediction  = (y_pred > 0.9).astype(int)
            
            
            if prediction ==0:
                prediction = "it's not"

            else:
                prediction = "it's Offensive!!"
                st.toast('Opsss!')    
            st.markdown('<div class="prediction">{} </div>'.format(prediction), unsafe_allow_html=True)


#'''Uploaded File page''' 
if selected == "Uploaded File":
    
    st.markdown('#')
    
    st.markdown('<h1 class="title">Upload Your File</h1>', unsafe_allow_html=True)
        
    st.markdown('#')
    uploaded_file = st.file_uploader("", type=['csv','xlsx'],accept_multiple_files=False,key="fileUploader")
    

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except:
            df = pd.read_excel(uploaded_file)
        
        progress_text = "Operation in progress. Please wait."
        my_bar = st.progress(0, text=progress_text)
        for percent_complete in range(100):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1, text=progress_text)
        
        col1,col2 = st.columns(2) 
        
        with col1:
            st.markdown('#')
            st.markdown('<div class="sliders">Original File</div>', unsafe_allow_html=True)
            df.reset_index(drop=True, inplace=True)
            st.write(df.head(4))
        
        with col2:
            st.markdown('#')
            st.markdown('<div class="sliders">Generate word cloud</div>', unsafe_allow_html=True)
            data = arabic_reshaper.reshape(df.to_string())
            data = get_display(data)
            
            wordcloud = WordCloud(font_path='arial',background_color='white', mode='RGB',width=2000,height=1000).generate(data)
            
            # Display the generated image:
            st.set_option('deprecation.showPyplotGlobalUse', False)
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            plt.show()
            st.pyplot()
        col5,col6,col7 = st.columns([1, 6, 1])
        with col6:
            # THE MODEL IS HEAR     
            X_train_processed = preprocessor.fit_transform(df)
        
            y_pred = model.predict(X_train_processed)
            result = np.where(y_pred > 0.5, 1, 0)
        
            st.markdown('#')
        
            #!!!!!!!!! PREDICTION     
            st.markdown('<div class="sliders">Predict File</div>', unsafe_allow_html=True)
            df.reset_index(drop=True, inplace=True)
            dfFanal = pd.DataFrame({'Class': result.flatten()})
            dfFanal = pd.concat([dfFanal, df],axis = 1)
            st.write(dfFanal.head(3))
        
            st.markdown('#')
            st.download_button('Download file',data=pd.DataFrame.to_csv(dfFanal,index=False), mime='text/csv')

#'''about page'''
if selected == "About":
    
    # gif from global file
    def load_lottieurl(url: str):
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    lottie_url = "https://lottie.host/b79d4a71-b8b4-465b-8bf1-fcaa0910449a/Mgxr39P1GP.json" 
    lottie_json = load_lottieurl(lottie_url)
    
    col1,col2 = st.columns(2) 
    
    with col1:
        st.markdown('<div class="about">Offensive Detection:</div>',unsafe_allow_html=True)
        st.markdown('<div class="about-info">is a system or algorithm designed to identify and flag language that could be considered offensive, harmful, inappropriate, or toxic. These detectors are commonly used in content moderation systems, chatbots, social media platforms, gaming environments, and other areas where user-generated content needs to be monitored.</div>',unsafe_allow_html=True)
    with col2:
        st_lottie(lottie_json,reverse=True, height=400, width=400, speed=0.5,)


if selected == "Creators":
    
    st.markdown('#')
    col1,col2,col3 = st.columns(3)
    with col1:
        pf1 = get_img_as_base64("RSN.png")
        st.markdown(f"""<div class='profile'>
                    <img src='data:image/png;base64,{pf1}' class='img'>
                    <br />
                    <br />
                    <div class='name'>
                    Ruwaidah Saud
                    </div>
                    Department of Science in Artificial Intelligence
                    <br />
                    Qassim University
                    <br />
                    461215489@qu.edu.sa 
                    </div>""",unsafe_allow_html=True)
    
    with col2:
        pf4 = get_img_as_base64("ys.png")
        st.markdown(f"""<div class='profile'>
                    <img src='data:image/png;base64,{pf4}' class='img'>
                    <br />
                    <br />
                    <div class='name'>
                    Yasir S. Alamri
                    </div>
                    Department of Science in Artificial Intelligence
                    <br />
                    Qassim University
                    <br />
                    461115823@qu.edu.sa 
                    </div>""",unsafe_allow_html=True)
    
    with col3:
        pf3 = get_img_as_base64("SA.png")
        st.markdown(f"""<div class='profile'>
                    <img src='data:image/png;base64,{pf3}' class='img'>
                    <br />
                    <br />
                    <div class='name'>
                    Sumaia Abdulwahab
                    </div>
                    Department of Science in Artificial Intelligence
                    <br />
                    Qassim University
                    <br />
                    461217096@qu.edu.sa 
                    </div>""",unsafe_allow_html=True)


