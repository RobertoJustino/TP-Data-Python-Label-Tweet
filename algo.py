import pandas as pd 
import numpy as np 
import seaborn as sb 
import matplotlib.pyplot as plt
import streamlit as st

df = pd.read_csv('labels.csv')

df.info()

number = st.number_input("Number of Rows to View")
st.dataframe(df.head(int(number)))
	
st.write("Columns Names")
st.write(df.columns.tolist())
    
dataLabel = df[['hate_speech','offensive_language','neither','class','tweet']]

dataLabel['tweetClean'] = dataLabel['tweet'].str.replace('\W', ' ')
dataLabel = dataLabel.head(5000)

X = dataLabel['tweetClean']
X
y = dataLabel['class']
y

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.3)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


from sklearn.pipeline import make_pipeline

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.multiclass import OneVsRestClassifier

from sklearn.svm import SVC

from stop_words import get_stop_words

clf = make_pipeline(

TfidfVectorizer(stop_words=get_stop_words('en')),
#transforms texts into a matrix of term-frequency times inverse document-frequency (tf-idf) values, suitable for machine learning

OneVsRestClassifier(SVC(kernel='linear', probability=True))
#For text classification, support vector machines (SVMs) are a reliable choice. As they are binary classifiers, we will use a One-Vs-Rest strategy, where for each #category an SVM is trained to separate this category from all others.

)


clf = clf.fit(X, y)

st.write("Score")
st.write(clf.score(x_train, y_train))

st.write("Predict")
st.write(clf.predict(x_test))

from joblib import dump, load
dump(clf, 'filename.joblib') 



