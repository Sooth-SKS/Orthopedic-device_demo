# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 11:33:17 2022

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



#st.set_page_config(page_title="Need of Simulation",layout = "wide")
#st.markdown("# Need of Simulation")

st.markdown("<hr/>", unsafe_allow_html=True)
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
    
