# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 11:31:29 2022

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
