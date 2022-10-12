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
import time
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots
warnings.filterwarnings("ignore")


st.set_page_config(layout="wide")



st.title('AI based assistance for design engineers to accelerate the product development process')
    
#st.subheader('A no-code AI platform which learns from historical FEA simulation data and predict the performance of a future design')
             
    

#st.title("AI based assistance for Design Engineers")

st.markdown("<h6 style='text-align: left; color: black;'>A no-code AI platform which learns from historical simulation/test data and predict the performance of a future design. </h6>", unsafe_allow_html=True)
#st.markdown("<h6 style='text-align: left; color:black;'> An end-to-end machine learning workflow that used simulated data from an EHR generates a risk-score against each visit. An overall risk-score is finally generated for each patient</h6>", unsafe_allow_html=True)
#st.write("An end-to-end machine learning workflow that used simulated data from an EHR and generates a risk-score against each visit. An overall risk-score is finally generated for each patient")
st.markdown("<hr/>", unsafe_allow_html=True)

st.markdown("<h5 style='text-align: left; color: black;'> Product background </h5>", unsafe_allow_html=True)

st.markdown(
"""
The product is selfdynamisable internal fixator (SIF), which is used in internal fixation of long thigh bones fractures (femur fractures).   
- The SIF comprises of a bar with anti-rotating screw in the dynamic unit on one end, with two clamps with corresponding locking screws and the trochanteric unit on the opposite end with two dynamic hip screws inside. 
- Similar to other fixation devices, the SIF represents the ultimate standard in internal fixation and in the healing of fractures without mechanical failure (e.g., bending of the bar or breaking of screws).
"""
)






fig_col1, fig_col2= st.columns(2)  


with fig_col1:
    image = Image.open('real product.jpg')
    st.image(image, width=500,caption='Different components of SIF device')
    
with fig_col2:
     image = Image.open('x ray image.jpg')
     st.image(image, width=500,caption='Radiographs of a right subtrochanteric femur fracture a) before treatment b) after treatment')
   


st.markdown("<hr/>", unsafe_allow_html=True)

st.markdown("<h5 style='text-align: left; color: black;'> Need of Simulation </h5>", unsafe_allow_html=True)



st.markdown(
"""
Mobility of the fractured segments is often beneficial for the formation of a callus, but it results in substantial loading of the applied fixation device, which may cause stability, strength, or durability related issues.    
- Structural analysis is employed to assess bone and fixator deformations, stresses, and strains, which are related to the fixator durability.
- For a known fixator configuration and position relative to the bone, structural analysis of bone-fixator systems is performed using the Finite Element Method (FEM).
- Using simulation data, an optimization study can be employed to find the optimum shape and dimensions of an existing fixation device. 
"""
)
    
    
fig_col1, fig_col2, fig_col3  = st.columns([6,1,6])  

with fig_col1:
    image = Image.open('FEA model SIF-Femur asssembly.jpg')
    st.image(image, width=450,caption='Finite element (FE) model of the femur–SIF assembly')
    
    
    
with fig_col3: 
    image = Image.open('stress field_1.png')
    st.image(image, width=400,caption='Stress field of the fixator (from FEA simulation)')
    



st.sidebar.markdown("## Input Paramters")
a = st.sidebar.slider('Bar length', min_value=100, max_value=250, step=10)
b = st.sidebar.slider('Bar diameter', min_value=8.0, max_value=10.0, step=0.1)
c = st.sidebar.slider('Bar end thickness', min_value=4.0, max_value=6.5, step=0.1)
d  = st.sidebar.slider('Radius trochanteric unit', min_value=3.0, max_value=10.0, step=0.1)
e  = st.sidebar.slider('Radius bar end', min_value=6.0, max_value=10.0, step=0.1)
f  = st.sidebar.slider('Clamp distance', min_value=1.0, max_value=28.0, step=0.5)


st.markdown(
"""
<style>
[data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
width: 300px;
}
[data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
width: 300px;
margin-left: -500px;
}
</style>
""",
unsafe_allow_html=True
)


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


with st.expander("Simulation dataset"):
    st.dataframe(df6)


    
st.markdown("<hr/>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: left; color: black;'> AI model training & prediction </h5>", unsafe_allow_html=True)
st.write("AI model is being trained from the past simulation data")

    



X=df6.values[:,:6]
y=df6.values[:,6:]
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size = .2)

input_data = np.array([a,b,c,d,e,f]).reshape(1,-1)

model = RandomForestRegressor()

st.spinner()
with st.spinner('Training AI model...'):
    
    
    time.sleep(1)
st.success('Taining is done !')


model.fit(X_train, y_train[:,0])
y_pred_1 = model.predict(input_data)
y_pred_test_1 = model.predict(X_test)


model = RandomForestRegressor()
model.fit(X_train, y_train[:,1])
y_pred_2 = model.predict(input_data)
y_pred_test_2 = model.predict(X_test)



model = RandomForestRegressor()
model.fit(X_train, y_train[:,2])
y_pred_3 = model.predict(input_data)
y_pred_test_3 = model.predict(X_test)



with st.expander("Design performance prediction"):
#st.markdown("<h6 style='text-align: left; color: black;'> Design performance prediction </h6>", unsafe_allow_html=True)
    st.write("The design performance parameters are total maximum deformation, equivalent stress, and fixator mass")
# create columns for the chars

    fig_col1, fig_col2, fig_col3 = st.columns(3)  
    with fig_col1:
    
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
        fig.update_layout(autosize=False, width=400, height=400)
        st.write(fig)
    
    with fig_col2:
    
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
        fig.update_layout(autosize=False, width=400, height=400)
        st.write(fig)
    
    
    with fig_col3:
    
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
        fig.update_layout(autosize=False, width=400, height=400)
        st.write(fig)
    

with st.expander("Design parameters sensitivity"):
    st.write("The importance score of the input design parameters")
    
    fig_col1, fig_col2, fig_col3  = st.columns([1,3,1])  

    with fig_col2:
        feature_importances = pd.DataFrame(model.feature_importances_,index = df6.columns[0:6],columns=['importance']).sort_values('importance', ascending=False)
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

with st.expander("AI model validation"):
    st.write("AI predictions are validated with the simulation data")
    fig = make_subplots(rows=1, cols=3,subplot_titles=("Total Deformation Maximum", "Equivalent Stress", "Fixator Mass"))

    fig.add_trace(
            go.Scatter(x =  y_test[:,0], y = y_pred_test_1, mode = 'markers',marker=dict(color='blue',size=12)),
            row=1, col=1)

    fig.add_trace(
            go.Scatter(x =  [5,12], y = [5,12], mode = 'lines',line=dict(color='black', width=2, dash='dash')),
            row=1, col=1)

    fig.update_xaxes(title_text="Simulation", range=[5,12],showgrid=False, row=1, col=1)
    fig.update_yaxes(title_text="Predicted", range=[5,12], showgrid=False, row=1, col=1)


    fig.add_trace(
            go.Scatter(x =  y_test[:,1], y = y_pred_test_2, mode = 'markers',marker=dict(color='blue',size=12)),
            row=1, col=2)

    fig.add_trace(
            go.Scatter(x =  [200,550], y = [200,550], mode = 'lines',line=dict(color='black', width=2, dash='dash')),
            row=1, col=2)


    fig.update_xaxes(title_text="Simulation", range=[200,550],showgrid=False, row=1, col=2)
    fig.update_yaxes(range=[200,550], showgrid=False, row=1, col=2)

    fig.add_trace(
            go.Scatter(x =  y_test[:,2], y = y_pred_test_3, mode = 'markers',marker=dict(color='blue',size=12)),
            row=1, col=3)

    fig.add_trace(
            go.Scatter(x =  [0.2,0.32], y = [0.2,0.32], mode = 'lines',line=dict(color='black', width=2, dash='dash')),
            row=1, col=3)


    fig.update_xaxes(title_text="Simulation", range=[0.2,0.32],showgrid=False, row=1, col=3)
    fig.update_yaxes(range=[0.2,0.32], showgrid=False, row=1, col=3)

    fig.update_layout(height=400, width=1000, showlegend=False)
    st.write(fig)
    
st.markdown("<hr/>", unsafe_allow_html=True)
with st.container():
   
    st.write("References: Korunovic et. al.,2019, In Silico Optimization of Femoral Fixator Position and Configuration by Parametric CAD Model, Materials 2019") 

def my_widget(key):
    st.subheader('Hello there!')
    return st.button("Click me " + key)

# This works in the main area
clicked = my_widget("first")

# And within an expander
my_expander = st.expander("Expand", expanded=True)
with my_expander:
    clicked = my_widget("second")

# AND in st.sidebar!
with st.sidebar:
    clicked = my_widget("third")
    



   