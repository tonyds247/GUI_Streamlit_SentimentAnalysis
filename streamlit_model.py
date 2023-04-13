# Import library
import streamlit as st
import pandas as pd
import numpy as np
import regex
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
#from wordcloud import WordCloud
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
st.write("#### Sentiment Analysis - Shopee E-commerce Plattform")


# PROCESS

# 1. Read Processed_Data
#@st.cache
df = pd.read_csv("shopee_final.csv", encoding='utf-8')


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

### Transform Data

#df['class'] = df['class'].apply(lambda x: 0 if x == 'negative' or x== 'neutral' else 1 )

#3. Buil Model
#vectorizer = dict(ngram_range=(1, 2), min_df=5, max_df=0.8)
#X_train, X_test, y_train, y_test = train_test_split(df['processed_text'], df['class'], test_size=0.2)
#pipeline = Pipeline(
    #[
        #("vect", CountVectorizer(**vectorizer)),
        #("tfidf", TfidfTransformer()),
        #('smt', RandomOverSampler()),
        #("clf", LogisticRegression(C=100, solver='newton-cg', penalty='l2'))
    #])
#model = pipeline.fit(X_train, y_train)

# Pickle
pkl_filename = "sentiment_bestmodel.pkl"  
pkl_count = "feature.pkl"  



#6. Load models 
# Đọc model
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
    st.write("""
    ##### 3 Classes - NaiveBayes
    """)
    st.code("Accuracy of model: 0.750074693228309")
    st.write("""
    ###### Confusion Matrix
    """)
    st.image("photo/p31.png")
    st.write("""
    ##### 3 Classes - LogisticRegression
    """)
    st.code("Params: maxIter=10, regParam=0.3")
    st.code("Accuracy of model: 0.7132097976225282")
    st.write("""
    ###### Confusion Matrix
    """)
    st.image("photo/p32.png")
    st.write("""
    #####
    Even though Naive Bayes gives better results than Logistic Regression, classification model has a poor performance
    """)

    st.write("""
    ##### 2 Classes - NaiveBayes
    """)
    st.code("Accuracy of model: 0.8256205630074483")
    st.write("""
    ###### Confusion Matrix
    """)
    st.image("photo/p21.png")
    st.write("""
    ##### 2 Classes - LogisticRegression
    """)
    st.code("Params: maxIter=10, regParam=0.3")
    st.code("Accuracy of model: 0.8546898330843918")
    st.write("""
    ###### Confusion Matrix
    """)
    st.image("photo/p22.png")
    st.write("""
    #####
    Based on the evaluation metrics, it appears that Logistic Regression performed better than Naive Bayes in terms of classification ability, despite the fact that the Accuracy score and confusion matrix yielded similar results to traditional Machine Learning models. Therefore, Logistic Regression may be a better option for this particular problem, despite the similarity in performance metrics to traditional models.
    """)




elif choice == 'New Prediction':
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
    st.write("""Please input a number, and a random data point will be selected !!
    """)
    num = st.number_input(label="", value=1)
    sub = st.button('Submit')

    if sub:
        df_index = df['processed_text'][num]
        df_index  = [df_index]
        st.write('You have just selected: ', df_index)
        tfidf2 = transformer.fit_transform(loaded_vec.fit_transform(df_index))

        text2 = tfidf2.toarray()
        result2 = sentiment_model.predict(text2)
        ng2= ""
        if result2 == 0:
            ng2 = 'Negative'
        else:
            ng2 = 'positive'
            
        st.write("The emotional tone of comment is: ",    (ng2))

    st.image("photo/icon.png")

    



