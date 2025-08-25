#This is a sample Python Script
import streamlit as st
import pandas as pd
import numpy as np
from os import path
import pickle # it is used to read .pkl file

st.title("Iris Dataset")
df_iris = pd.read_csv(path.join("Data","iris.csv"))
# filepath = Root/Data/iris.csv
st.write(df_iris)
# Plotting a scatter plot using the data
st.scatter_chart(df_iris[['sepal_length', 'sepal_width']])

# petal_length = st.number_input("Please choose a Petal length:", max=6.9, min=1)
# petal_width = st.slider("Please choose a Petal width:")
# sepal_length = st.slider("Please choose a Sepal length:")
# sepal_width = st.slider("Please choose a Sepal width:")

st.title("Flower Species Predictor")
petal_length = st.number_input("Please choose a Petal length:", placeholder="Please enter a valid number between 1.0 and 6.9", min_value=1.0, max_value=6.9, value=None)
petal_width = st.number_input("Please choose a Petal width:", placeholder="Please enter a valid number between 0.1 and 2.5", min_value=0.1, max_value=2.5, value=None)
sepal_length = st.number_input("Please choose a Sepal length:", placeholder="Please enter a valid number between 4.3 and 7.9", min_value=4.3, max_value=7.9, value=None)
sepal_width = st.number_input("Please choose a Sepal width:", placeholder="Please enter a valid number between 2.0 and 4.4", min_value=2.0, max_value=4.4, value=None)

# prepare the dataframe for prediction
user_input = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
st.write(user_input)

# using the .pkl file, creating an ML model named 'iris_predictor'
model_path = path.join("Model","iris_classifier.pkl")
with open(model_path, "rb") as file:
    iris_predictor = pickle.load(file)

# st.write(iris_predictor)

species = {0:'setosa', 1:'versicolor', 2:'virginica'}

if st.button("Predict Species"):
      if ((petal_length==None)or(petal_width==None)or(sepal_length==None)or(sepal_width==None)):
          # will be executed when any of the values is not entered properly
          st.write("Please enter a valid number")

      else:
          # prediction can be done here
          predicted_species = iris_predictor.predict(user_input)
          # predicted species[0] will give us the value in the df
          # we use that value to find the corresponding species from the dictionary 'species'
          st.write("The species is:", species[predicted_species[0]])