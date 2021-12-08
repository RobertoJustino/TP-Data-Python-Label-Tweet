import pandas as pd 
import numpy as np 
import seaborn as sb 
import matplotlib.pyplot as plt
import streamlit as st

df = pd.read_csv('labels.csv')

df.info()

if st.checkbox("Show DataSet"):
	number = st.number_input("Number of Rows to View")
	st.dataframe(df.head(int(number)))
	
if st.button("Columns Names"):
    st.write(df.columns.tolist())
    
dataLabel = df[['hate_speech','offensive_language','neither','class','tweet']]

dataLabel['tweet'] = dataLabel['tweet'].str.replace('\W', '')
dataLabel

X = dataLabel['tweet']
X
y = dataLabel['class']
y

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.3)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
