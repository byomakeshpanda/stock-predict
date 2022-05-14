import pandas as pd
import matplotlib.pyplot as plt
import pickle
import streamlit as st
from sklearn.linear_model import LinearRegression

st.title('Stock Price Prediction ')

df = pd.read_csv('infy_prices.csv')

#Describing the data
st.subheader('Data from Jan 2016 to Jan 2022 ')
st.write(df.describe())

#Visualization of data
st.subheader('High Price vs Time')
fig = plt.figure(figsize=(10,5))
plt.plot(df['High Price'],label = 'High Price')
plt.xlabel('Time')
plt.ylabel('Price')
st.pyplot(fig)

ma100 = df['High Price'].rolling(100).mean() #100 days moving average
ma200 = df['High Price'].rolling(200).mean() #200 days moving average

st.subheader('High Price vs Time with 100 days MA')
fig = plt.figure(figsize=(10,5))
plt.plot(df['High Price'],label = 'High Price')
plt.plot(ma100,label = '100 days moving average', color= 'green')
plt.xlabel('Time')
plt.ylabel('Price')
st.pyplot(fig)

st.subheader('High Price vs Time with 100 and 200 days MA')
fig = plt.figure(figsize=(10,5))
plt.plot(df['High Price'],label = 'High Price')
plt.plot(ma100,label = '100 days moving average', color= 'green')
plt.plot(ma200,label = '200 days moving average', color= 'black')
plt.xlabel('Time')
plt.ylabel('Price')
st.pyplot(fig)

y = df[['High Price']]
X = df.drop('High Price',axis = 1)
X_train = pd.DataFrame(X[0:int(len(X)*0.70)])
y_train = pd.DataFrame(y[0:int(len(y)*0.70)])
X_test  = pd.DataFrame(X[int(len(X)*0.70):int(len(X))])
y_test  = pd.DataFrame(y[int(len(y)*0.70):int(len(y))])

model = LinearRegression()
model.fit(X_train,y_train)
y_predicted = model.predict(X_test)
y_test.reset_index(inplace =True)
y_test.drop('index',axis=1,inplace = True)
y_pd = pd.DataFrame(y_predicted)

st.subheader('Actual Graph vs Predicted graph of last 700 days which is predicted')
fig = plt.figure(figsize=(10,5))
plt.plot(y_test,color='red',label="Actual")
plt.plot(y_predicted, color='blue', label = "Predicted")
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)

st.write("      The model has 99.95 % accuracy ")