# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 14:25:17 2022

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
import time
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots
warnings.filterwarnings("ignore")


st.set_page_config(layout="wide")


def add_logo(logo_path, width, height):
    """Read and return a resized logo"""
    logo = Image.open(logo_path)
    modified_logo = logo.resize((width, height))
    return modified_logo

my_logo = add_logo(logo_path="Soosthsayer_logo.png", width=280, height=100)
st.sidebar.image(my_logo)


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
#df6.head()






X=df6.values[:,:6]
y=df6.values[:,6:]
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size = .2)



model1 =RandomForestRegressor()
model2 = RandomForestRegressor()
model3 = RandomForestRegressor()



model1.fit(X_train, y_train[:,0])
y_pred_test_1 = model1.predict(X_test)


model2.fit(X_train, y_train[:,1])
y_pred_test_2 = model2.predict(X_test)


model3.fit(X_train, y_train[:,2])
y_pred_test_3 = model3.predict(X_test)



#with st.expander("Design performance prediction"):
#st.markdown("<h6 style='text-align: left; color: black;'> Design performance prediction </h6>", unsafe_allow_html=True)
#st.write("Based on the designer inputs, the design performance can be predicted. Here, the design performance parameters are total maximum deformation, equivalent stress, and fixator mass")
# create columns for the chars
#st.markdown("<hr/>", unsafe_allow_html=True)

fig_col0, fig_col1, fig_col2, fig_col3 = st.columns([0.8,1,1,1])
with fig_col0:
        st.markdown("Input Parameters")
        a = st.slider('Bar length', min_value=100, max_value=250, step=10)
        b = st.slider('Bar diameter', min_value=8.0, max_value=10.0, step=0.1)
        c = st.slider('Bar end thickness', min_value=4.0, max_value=6.5, step=0.1)
        d  = st.slider('Radius trochanteric unit', min_value=3.0, max_value=10.0, step=0.1)
        e  = st.slider('Radius bar end', min_value=6.0, max_value=10.0, step=0.1)
        f  = st.slider('Clamp distance', min_value=1.0, max_value=28.0, step=0.5)   
        input_data = np.array([a,b,c,d,e,f]).reshape(1,-1)

       
   
   
   
with fig_col1:
        y_pred_1 = model1.predict(input_data)
        fig = go.Figure(go.Indicator(
                domain = {'x': [0, 1], 'y': [0, 1]},
                value = round(y_pred_1[0],2),
                mode = "gauge+number+delta",
                title = {'text': "Total Deformation Maximum"},
                delta = {'reference': 18, 'increasing': {'color': "red"},'decreasing': {'color': "green"}},
                gauge = {'axis': {'range': [0, 20]},
                                  'bar': {'color': "black"},
                                  'steps' : [
                                          {'range': [0, 10], 'color': "lightgreen "},
                                          {'range': [10, 15], 'color': "yellow"},
                                          {'range': [15, 20], 'color': "red"}],
                                          'threshold' : {'line': {'color': "darkblue", 'width': 6}, 'thickness': 0.75, 'value': 18}}))
        fig.update_layout(autosize=False, width=350,height=400)
        st.write(fig)
    
with fig_col2:
        y_pred_2 = model2.predict(input_data)
        fig = go.Figure(go.Indicator(
                domain = {'x': [0, 1], 'y': [0, 1]},
                value = round(y_pred_2[0],2),
                mode = "gauge+number+delta",
                title = {'text': "Equivalent Stress"},
                delta = {'reference': 600, 'increasing': {'color': "red"},'decreasing': {'color': "green"}},
                gauge = {'axis': {'range': [0, 700]}, 'bar': {'color': "black"},
                                  'steps' : [
                                          {'range': [0, 200], 'color': "lightgreen "},
                                          {'range': [200, 500], 'color': "yellow"},
                                          {'range': [500, 800], 'color': "red"}],
                                          'threshold' : {'line': {'color': "darkblue", 'width': 4}, 'thickness': 0.75, 'value': 600}}))
        fig.update_layout(autosize=False, width=350, height=400)
        st.write(fig)
    
    
with fig_col3:
        y_pred_3 = model3.predict(input_data)
        fig = go.Figure(go.Indicator(
                domain = {'x': [0, 1], 'y': [0, 1]},
                value = round(y_pred_3[0],2),
                mode = "gauge+number+delta",
                title = {'text': "Fixator Mass"},
                delta = {'reference': 0.25, 'increasing': {'color': "red"},'decreasing': {'color': "green"}},
                gauge = {'axis': {'range': [0, 0.5]},
                                  'bar': {'color': "black"},
                                  'steps' : [
                                          {'range': [0, 0.1], 'color': "lightgreen "},
                                          {'range': [0.1, 0.3], 'color': "yellow"},
                                          {'range': [0.3, 0.6], 'color': "red"}],
                                          'threshold' : {'line': {'color': "darkblue", 'width': 4}, 'thickness': 0.75, 'value': 0.25}}))
        fig.update_layout(autosize=False,width=350, height=400)
        st.write(fig)
        
    

with st.expander("Design parameters sensitivity"):
    st.write("The importance score of the input design parameters")
    
    fig_col1, fig_col2, fig_col3  = st.columns([1,3,1])  

    with fig_col2:
        feature_importances = pd.DataFrame(model2.feature_importances_,index = df6.columns[0:6],columns=['importance']).sort_values('importance', ascending=False)
        num = feature_importances.shape[0]
        ylocs = np.linspace(1,num,num)
        values_to_plot = feature_importances[:num].values.ravel()[::-1]
        feature_labels = list(feature_importances[:num].index)[::-1]
        #plt.figure(num=None, facecolor='w', edgecolor='k');
        plt.barh(ylocs, values_to_plot, align = 'center', height = 0.8)
        plt.ylabel('Features')
        plt.xlabel('Featur importance score')
        plt.yticks(ylocs, feature_labels)
        st.pyplot(plt)