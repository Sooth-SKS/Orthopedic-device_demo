# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 15:38:55 2022

@author: susym
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, make_scorer, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import collections
import warnings
from PIL import Image
import plotly.graph_objects as go
warnings.filterwarnings("ignore")



st.set_page_config(layout="wide")

st.title("AI based assistance for Design Engineers")

image = Image.open('parametric model.jpg')
st.image(image, width=500, caption='Orthopedic device – internal fixator, used in subtrochanteric fractures of thighbone (femur)')


st.sidebar.markdown("## Input Paramters")
a = st.sidebar.slider('Bar length', min_value=100, max_value=250, step=10)
b = st.sidebar.slider('Bar diameter', min_value=8.0, max_value=10.0, step=0.1)
c = st.sidebar.slider('Bar end thickness', min_value=4.0, max_value=6.5, step=0.1)
d  = st.sidebar.slider('Radius trochanteric unit', min_value=3.0, max_value=10.0, step=0.1)
e  = st.sidebar.slider('Radius bar end', min_value=6.0, max_value=10.0, step=0.1)
f  = st.sidebar.slider('Clamp distance', min_value=1.0, max_value=28.0, step=0.5)


df6=pd.read_csv("./DOE6.csv", 
                skiprows=4, 
                names=['Name',
                       'Bar length',
                       'Bar diameter',
                       'Bar end thickness',
                       'Radius trochanteric unit',
                       'Radius bar end',
                       'Clamp distance',
                       'Total Deformation Maximum',
                       'Equivalent Stress',
                       'P9',
                       'P10',
                       'P11',
                       'Fixator Mass'], 
                usecols=['Bar length',
                         'Bar diameter',
                         'Bar end thickness',
                         'Radius trochanteric unit',
                         'Radius bar end',
                         'Clamp distance',
                         'Total Deformation Maximum',
                         'Equivalent Stress',
                         'Fixator Mass'])
df6.head()


X=df6.values[:,:6]
y=df6.values[:,6:]
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size = .2)

input_data = np.array([a,b,c,d,e,f]).reshape(1,-1)

model = RandomForestRegressor()
model.fit(X_train, y_train[:,0])
y_pred_1 = model.predict(input_data)

model = RandomForestRegressor()
model.fit(X_train, y_train[:,1])
y_pred_2 = model.predict(input_data)

model = RandomForestRegressor()
model.fit(X_train, y_train[:,2])
y_pred_3 = model.predict(input_data)


st.write("Machine Learning Prediction for design performance")


# create columns for the chars

fig_col1, fig_col2, fig_col3 = st.columns(3)  
with fig_col1:
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = round(y_pred_1[0],2),
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Total Deformation Maximum"}))
    
    fig.update_layout(
        autosize=False,
        width=300,
        height=300,
  
    )
    st.write(fig)
    
with fig_col2:
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = round(y_pred_2[0],2),
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Equivalent Stress"}))
    
    fig.update_layout(
        autosize=False,
        width=300,
        height=300,
  
    )
    st.write(fig)
    
with fig_col3:
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = round(y_pred_3[0],2),
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Fixator Mass"}))
    
    fig.update_layout(
        autosize=False,
        width=300,
        height=300,
  
    )
    st.write(fig)




