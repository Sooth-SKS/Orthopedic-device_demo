# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 11:35:21 2022

@author: susym
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import time
from PIL import Image
import plotly.graph_objects as go
from plotly.subplots import make_subplots
#warnings.filterwarnings("ignore")


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


X=df6.values[:,:6]
y=df6.values[:,6:]
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size = .2)

model1 =RandomForestRegressor()
model2 = RandomForestRegressor()
model3 = RandomForestRegressor()

model1.fit(X_train, y_train[:,0])
model2.fit(X_train, y_train[:,1])
model3.fit(X_train, y_train[:,2])

y_pred_test_1 = model1.predict(X_test)
y_pred_test_2 = model2.predict(X_test)
y_pred_test_3 = model3.predict(X_test)

    

result = st.sidebar.button("Start training")

if result:

    st.spinner()
    with st.spinner('Training AI model...'):
    
        time.sleep(5)
        st.balloons()

        st.sidebar.success('Taining is done !')
        
    valid = st.sidebar.radio(label = "", options = ['Validation plots', 'Error analysis'])
    #session_state = get(a=0, b=0)

    if valid == 'Validation plots':
            st.title("Validation against the test simulation data")   
            st.markdown("<hr/>", unsafe_allow_html=True)    
    
            fig = make_subplots(rows=1, cols=3,subplot_titles=("Total Deformation Maximum", "Equivalent Stress", "Fixator Mass"))

            fig.add_trace(
                    go.Scatter(x =  y_test[:,0], y = y_pred_test_1, mode = 'markers',marker=dict(color='blue',size=8)),
                    row=1, col=1)

            fig.add_trace(
                    go.Scatter(x =  [5,12], y = [5,12], mode = 'lines',line=dict(color='black', width=2, dash='dash')),
                    row=1, col=1)

            fig.update_xaxes(title_text="Simulation", range=[5,12],showgrid=False, row=1, col=1)
            fig.update_yaxes(title_text="Predicted", range=[5,12], showgrid=False, row=1, col=1)


            fig.add_trace(
                    go.Scatter(x =  y_test[:,1], y = y_pred_test_2, mode = 'markers',marker=dict(color='blue',size=8)),
                    row=1, col=2)

            fig.add_trace(
                    go.Scatter(x =  [200,550], y = [200,550], mode = 'lines',line=dict(color='black', width=2, dash='dash')),
                    row=1, col=2)


            fig.update_xaxes(title_text="Simulation", range=[200,550],showgrid=False, row=1, col=2)
            fig.update_yaxes(range=[200,550], showgrid=False, row=1, col=2)

            fig.add_trace(
                    go.Scatter(x =  y_test[:,2], y = y_pred_test_3, mode = 'markers',marker=dict(color='blue',size=8)),
                    row=1, col=3)

            fig.add_trace(
                    go.Scatter(x =  [0.2,0.32], y = [0.2,0.32], mode = 'lines',line=dict(color='black', width=2, dash='dash')),
                    row=1, col=3)


            fig.update_xaxes(title_text="Simulation", range=[0.2,0.32],showgrid=False, row=1, col=3)
            fig.update_yaxes(range=[0.2,0.32], showgrid=False, row=1, col=3)

            fig.update_layout(height=400, width=1000, showlegend=False)
            st.write(fig)
        
    else:
    
      
            fig_col0, fig_col1, fig_col2 = st.columns(3)
            #with fig_col0:
                #st.markdown('Avg. error(Total Deformation Maximum): ' + str(round(a,2)))
        
            with fig_col1:
                #st.markdown('Avg. error(Equivalent Stress): ' + str(round(b,2)))
                st.title("Error with respect to the test simulation data")
        
            #with fig_col2:
                #st.markdown('Avg. error(Fixator Mass): ' + str(round(c,2)))




with st.expander("Error Analysis"):
            
            st.title("Error with respect to the test simulation data")
            st.markdown("<hr/>", unsafe_allow_html=True)
            con_1 = np.concatenate((y_test[:,0].reshape(-1,1), y_pred_test_1.reshape(-1,1)), axis=1)
            con_2 = np.concatenate((y_test[:,1].reshape(-1,1), y_pred_test_2.reshape(-1,1)), axis=1)
            con_3 = np.concatenate((y_test[:,2].reshape(-1,1), y_pred_test_3.reshape(-1,1)), axis=1)

            df_val_1 = pd.DataFrame(con_1, columns=['test', 'pred'])
            df_val_2 = pd.DataFrame(con_2, columns=['test', 'pred'])
            df_val_3 = pd.DataFrame(con_3, columns=['test', 'pred'])

            df_val_1['% Error(Total Deformation Maximum)'] = abs(df_val_1['pred']-df_val_1['test'])*100/df_val_1['test'] 
            df_val_2['% Error(Equivalent Stress)'] = abs(df_val_2['pred']-df_val_2['test'])*100/df_val_2['test'] 
            df_val_3['% Error(Fixator Mass)'] = abs(df_val_3['pred']-df_val_3['test'])*100/df_val_3['test'] 

            df_val_all = pd.concat([df_val_1,df_val_2,df_val_3],axis = 1).drop(['test', 'pred'],axis=1)
            df_X_test = pd.DataFrame(X_test[0:6], columns=['Bar length','Bar diameter','Bar end thickness','Radius trochanteric unit','Radius bar end','Clamp distance'])

            df_val_error = pd.merge(df_X_test,df_val_all,left_index=True, right_index=True)

            a = df_val_error['% Error(Total Deformation Maximum)'].mean()
            b = df_val_error['% Error(Equivalent Stress)'].mean()
            c = df_val_error['% Error(Fixator Mass)'].mean()
            st.dataframe(df_val_error)
    



    
