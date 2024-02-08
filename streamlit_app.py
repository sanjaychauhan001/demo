import streamlit as st 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import plotly.express as px
df = pd.read_csv("Iris.csv") 



st.sidebar.header("Description")
st.sidebar.subheader("The Iris Dataset contains four features (length and width of sepals and petals) of 50 samples of three species of Iris (Iris setosa, Iris virginica and Iris versicolor). These measures were used to create a linear discriminant model to classify the species.")
st.title("IRIS FLOWER DASHBOARD")
st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("pie chart of species")
    fig1, ax1 = plt.subplots()
    ax1.pie(x=df['Species'].value_counts().values, labels=df['Species'].value_counts().index, autopct="%0.2f%%")
    st.pyplot(fig=fig1)

with col2:
    st.subheader("count plot of species")
    fig2, ax2 = plt.subplots()
    ax2.bar(x=df['Species'].value_counts().index, height=df['Species'].value_counts().values)
    st.pyplot(fig=fig2)    


st.subheader("line chart of species")
st.line_chart(df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']])    

col3, col5 = st.columns(2)

with col3:
    st.subheader("scatter plot of sepal length VS sepal width")
    fig3, ax3 = plt.subplots()
    ax3.scatter(x=df['SepalLengthCm'],y=df['SepalWidthCm'])
    st.pyplot(fig=fig3)

st.subheader("3d scatter of sepal_length vs sepal_width vs petal_length")
figure = px.scatter_3d(x=df['SepalLengthCm'],y=df['SepalWidthCm'],z=df['PetalLengthCm'])
st.plotly_chart(figure)


with col5:
    st.subheader("scatterplot of petal length VS petal width")
    fig5, ax5 = plt.subplots()
    ax5.scatter(x=df['PetalLengthCm'],y=df['PetalWidthCm'])
    st.pyplot(fig=fig5)    