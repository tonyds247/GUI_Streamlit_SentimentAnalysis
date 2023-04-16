# Import library
import streamlit as st
import pandas as pd
import numpy as np
import regex
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import json
from string import punctuation 
from underthesea import word_tokenize, pos_tag, sent_tokenize
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split 
from imblearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

st.set_option('deprecation.showPyplotGlobalUse', False)


#--------------
# GUI
st.image('photo/shopee_11zon.jpg')
st.title("Data Science Capstone Project ")
st.write("#### Project name: Sentiment Analysis - Shopee E-commerce Plattform")
st.write("#### by Tung Nguyen ")
st.write("---------------------------------------------------")


# PROCESS

# 1. Read Processed_Data
#@st.cache
df = pd.read_csv("new_shopee.csv", encoding='utf-8')
df3a= pd.read_json("json/3a.json")
df3b= pd.read_json("json/3b.json")
df2a= pd.read_json("json/2a.json")
df2b= pd.read_json("json/2b.json")
df_1 = df['processed_text'].sample(frac=0.00001, random_state=1)
df_1.reset_index(drop=True, inplace=True)
df_2 = df['processed_text'].sample(frac=0.00001, random_state=2)
df_2.reset_index(drop=True, inplace=True)
df_3 = df['processed_text'].sample(frac=0.00001, random_state=3)
df_3.reset_index(drop=True, inplace=True)
df_4 = df['processed_text'].sample(frac=0.00001, random_state=4)
df_4.reset_index(drop=True, inplace=True)

# 2. Clean Text
### clean user_input 
def pre_process(text):
    user_input = re.sub('[\.\:\,\-\+\d\!\%\...\"\*\>\<\^\&\/\[\]\(\)\=\~\#]', ' ', text)
    user_input = regex.sub(r'\s+', ' ', user_input).strip()
    
    def loaddicchar():
            uniChars = "àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệđìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆĐÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴÂĂĐÔƠƯ"
            unsignChars = "aaaaaaaaaaaaaaaaaeeeeeeeeeeediiiiiooooooooooooooooouuuuuuuuuuuyyyyyAAAAAAAAAAAAAAAAAEEEEEEEEEEEDIIIOOOOOOOOOOOOOOOOOOOUUUUUUUUUUUYYYYYAADOOU"

            dic = {}
            char1252 = 'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ'.split(
                '|')
            charutf8 = "à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ".split(
                '|')
            for i in range(len(char1252)):
                dic[char1252[i]] = charutf8[i]
            return dic
         
        # 
    def covert_unicode(txt):
            dicchar = loaddicchar()
            return regex.sub(
                r'à|á|ả|ã|ạ|ầ|ấ|ẩ|ẫ|ậ|ằ|ắ|ẳ|ẵ|ặ|è|é|ẻ|ẽ|ẹ|ề|ế|ể|ễ|ệ|ì|í|ỉ|ĩ|ị|ò|ó|ỏ|õ|ọ|ồ|ố|ổ|ỗ|ộ|ờ|ớ|ở|ỡ|ợ|ù|ú|ủ|ũ|ụ|ừ|ứ|ử|ữ|ự|ỳ|ý|ỷ|ỹ|ỵ|À|Á|Ả|Ã|Ạ|Ầ|Ấ|Ẩ|Ẫ|Ậ|Ằ|Ắ|Ẳ|Ẵ|Ặ|È|É|Ẻ|Ẽ|Ẹ|Ề|Ế|Ể|Ễ|Ệ|Ì|Í|Ỉ|Ĩ|Ị|Ò|Ó|Ỏ|Õ|Ọ|Ồ|Ố|Ổ|Ỗ|Ộ|Ờ|Ớ|Ở|Ỡ|Ợ|Ù|Ú|Ủ|Ũ|Ụ|Ừ|Ứ|Ử|Ữ|Ự|Ỳ|Ý|Ỷ|Ỹ|Ỵ',
                lambda x: dicchar[x.group()], txt)
                
    user_input = covert_unicode(user_input)
    user_input = word_tokenize(user_input, format="text")
    user_input = user_input.lower()
    
    return user_input

# Pickle
pkl_filename = "sentiment_bestmodel.pkl"  
pkl_count = "feature.pkl"  

# Load models 
with open(pkl_filename, 'rb') as file:  
    sentiment_model= pickle.load(file)

#Load countvec
transformer = TfidfTransformer()
loaded_vec = CountVectorizer(decode_error="replace",vocabulary=pickle.load(open("feature.pkl", "rb")))


#Caching the model for faster loading
#@st.cache

# GUI
menu = ["Business Objective","Model Selection and Result", "New Prediction"]
choice = st.sidebar.selectbox('Menu', menu)


if choice == 'Business Objective':    
    st.subheader("Business Objective")
    st.write("""
    ###### Sentiment analysis is the process of analyzing digital text to determine if the emotional tone of the message is positive, negative, or neutral.
    - The project consists of 6 steps:
        + Preprocessing: Exploratory Data Analysis and text cleaning
        + Model Pre-selection: Using Lazy Predict to identify potential machine learning models
        + Model selection with Machine Learning (for 3 classes): Applying selected machine learning models from Lazy Predict, choosing the best model, and tuning hyperparameters
        + Model selection with Machine Learning (for 2 classes): Using SMOTE as a resampling technique, applying selected machine learning models from Lazy Predict, choosing the best model, and tuning hyperparameters
        + Model selection with Pyspark (for 3 classes): Comparing Naive Bayes and Logistic Regression models
        + Model selection with Pyspark (for 2 classes): Comparing Naive Bayes and Logistic Regression models
    - The result will be discussed in the next page (Model Selection and Result)
    - The goal is to analyze and understand user sentiment towards a particular product, service, or topic, which can help businesses improve their offerings and enhance customer satisfaction.  
      """)  
    st.write("""###### => Problem/ Requirement: Use Machine Learning algorithms in Python for sentiment comments classification.""")
    st.image("photo/bo.jpg")

elif choice == "Model Selection and Result":
    st.write("""
    ### 
    EDA
    """)
    st.dataframe(df[['processed_text', 'class']].head(5))
    st.dataframe(df['class'].value_counts())
    st.write("""
    ###### 
    - The dataset can be divided into 3 classes based on customer ratings: 1-2 (negative), 3 (neutral), and 4-5 (positive) However, ratings 1-3 can be seen as areas that require improvement
    - Data set seems to be imbalanced   
    """)

    st.write("""
    #### Pre-selection with Lazy Predict
    - Based on Accuracy from Lazypredict, the following models will be shortlisted in building model
    - 3 classes
        + RandomForestClassifier
        + KNeighborsClassifier
        + LogisticRegression
    - 2 Classes
        + ExtraTreesClassifier
        + XGBClassifier
        + LogisticRegression
        + KNeighborsClassifier
    """)
    ### RESULT
    st.write("""
    #### Result
    - For 3 classes, applying resampling did not improve results, while for 2 classes, resampling using SMOTE improved results The final metrics show that normal processing for 3 classes resulted in poor performance, while processing for 2 classes performed better.
    """)
    st.write("""
    ###### Machine Learning Model - 3 Classes
    """)
    st.image("photo/31.png")
    st.write("""
    ###### 
    - Logistic Regression is best model 
    """)
    st.write("""
    ###### Tuning Model
    """)
    st.code("Best Params: {'C': 100, 'penalty': 'l2', 'solver': 'newton-cg'}")
    st.write("""
    ###### Final Result
    """)
    st.image("photo/32.png")
    st.image("photo/33.png")
    st.write("""
    ###### The model's performance is poor
    """)
    ### 2 CLASS
    st.write("""
    ###### Machine Learning Model -2 Classes
    """)
    st.image("photo/211.png")
    st.write("""
    ###### 
    - Logistic Regression is best model 
    """)
    st.write("""
    ###### Tuning Model
    """)
    st.code("Best Params: {'C': 1.0, 'penalty': 'l2', 'solver': 'lbfgs'}")
    st.write("""
    ###### Final Result
    """)
    st.image("photo/22.png")
    st.image("photo/23.png")
    st.write("""
    ######
    - The model performs well for the two classes, with a classification accuracy of over 80%.
    """)

    ### Pyspark
    
    st.write("""
    ### Pyspark
    PySpark's MLlib library to train a Logistic Regression and Naive Bayes model
    """)
    st.image("photo/pyspark.jpg")
    ### 3C
    st.write("""
    ##### 3 Classes - NaiveBayes
    """)
    st.code("Accuracy of model: 0.750074693228309")
    st.write("""
    ###### Confusion Matrix
    """)
    st.dataframe(df3a['3-Class Confusion Matrix'])

    st.write("""
    ##### 3 Classes - LogisticRegression
    """)
    st.code("Params: maxIter=10, regParam=0.3")
    st.code("Accuracy of model: 0.7132097976225282")
    st.write("""
    ###### Confusion Matrix
    """)
    st.dataframe(df3b['3-Class Confusion Matrix'])
    st.write("""
    #####
    Even though Naive Bayes gives better results than Logistic Regression, classification model has a poor performance
    """)

    ###2C
    st.write("""
    ##### 2 Classes - NaiveBayes
    """)
    st.code("Accuracy of model: 0.8256205630074483")
    st.write("""
    ###### Confusion Matrix
    """)
    st.dataframe(df2a['2-Class Confusion Matrix'])

    st.write("""
    ##### 2 Classes - LogisticRegression
    """)
    st.code("Params: maxIter=10, regParam=0.3")
    st.code("Accuracy of model: 0.8546898330843918")
    st.write("""
    ###### Confusion Matrix
    """)
    st.dataframe(df2b['2-Class Confusion Matrix'])
    st.write("""
    #####
    Based on the evaluation metrics, it appears that Logistic Regression performed better than Naive Bayes in terms of classification ability, despite the fact that the Accuracy score and confusion matrix yielded similar results to traditional Machine Learning models. Therefore, Logistic Regression may be a better option for this particular problem, despite the similarity in performance metrics to traditional models.
    """)




elif choice == 'New Prediction':
    st.write("""
    ####
    - Used model: Logistic Regression
    - Output: 
        + Negative: Rating 1,2,3
        + Positive: Rating 4,5
    """)
    st.write("""
    Params tuned
    """)
    st.code("Best Params: {'C': 1.0, 'penalty': 'l2', 'solver': 'lbfgs'}")
    st.code('Accuracy score for the best model on the training data: 73.1 ')
    st.code('Accuracy score for the best model on the training data: 73.2 ')
    ### result:


    st.subheader("Data Entry")
    st.write('White a comment')
    cmt= st.text_area(label= "")
    submit = st.button('Send')

    if submit:
        user_input =  [pre_process(cmt)]
        st.write('text after processed: ', user_input)
        tfidf1 = transformer.fit_transform(loaded_vec.fit_transform(user_input))

        text = tfidf1.toarray()
        result = sentiment_model.predict(text)
        ng = ""
        if result == 0:
            ng = 'Negative'
        else:
            ng = 'positive'
            
        st.write("The emotional tone of comment is: ",    (ng))
    
    ## 
    st.subheader("Data Selection")
    option = st.selectbox('Please select your Dataset',("Dateset 01", "Dateset 02", "Dateset 03", "Dateset 04"))
    st.write('You selected: ', option )
    submit1 = st.button('Predict')
    if submit1:
        if option == "Dateset 01":
            st.dataframe(df_1)
            tfidf2_1 = transformer.fit_transform(loaded_vec.fit_transform(df_1))

            ###
            result_1 = sentiment_model.predict(tfidf2_1)
            df_r1 = pd.DataFrame(result_1)
            df_r1['Prediction'] = df_r1
        
            df_r1= df_r1.applymap(lambda x: 'Negative' if x == 0 else 'Positive')
            r1= pd.concat([df_1,df_r1], axis=1)
            r1 = r1[['processed_text', 'Prediction']]
            st.write("""
            ##### Result
            """)
            st.write(r1)
        if option == "Dateset 02":
            st.dataframe(df_2)
            tfidf2_2 = transformer.fit_transform(loaded_vec.fit_transform(df_2))

            ###
            result_2 = sentiment_model.predict(tfidf2_2)
            df_r2 = pd.DataFrame(result_2)
            df_r2['Prediction'] = df_r2
        
            df_r2 = df_r2.applymap(lambda x: 'Negative' if x == 0 else 'Positive')
            r2= pd.concat([df_2,df_r2], axis=1)
            r2 = r2[['processed_text', 'Prediction']]
            st.write("""
            ##### Result
            """)
            st.write(r2)
        if option == "Dateset 03":
            st.dataframe(df_3)
            tfidf2_3 = transformer.fit_transform(loaded_vec.fit_transform(df_3))

            ###
            result_3 = sentiment_model.predict(tfidf2_3)
            df_r3 = pd.DataFrame(result_3)
            df_r3['Prediction'] = df_r3
        
            df_r3 = df_r3.applymap(lambda x: 'Negative' if x == 0 else 'Positive')
            r3= pd.concat([df_3,df_r3], axis=1)
            r3 = r3[['processed_text', 'Prediction']]
            st.write("""
            ##### Result
            """)
            st.write(r3)
        if option == "Dateset 04":
            st.dataframe(df_4)
            tfidf2_4 = transformer.fit_transform(loaded_vec.fit_transform(df_4))

            ###
            result_4 = sentiment_model.predict(tfidf2_4)
            df_r4 = pd.DataFrame(result_4)
            df_r4['Prediction'] = df_r4
        
            df_r4 = df_r4.applymap(lambda x: 'Negative' if x == 0 else 'Positive')
            r4= pd.concat([df_4,df_r4], axis=1)
            r4 = r4[['processed_text', 'Prediction']]
            st.write("""
            ##### Result
            """)
            st.write(r4)
    st.image("photo/icon.jpg")

    



