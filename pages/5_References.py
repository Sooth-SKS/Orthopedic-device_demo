# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 11:21:51 2022

@author: susym
"""
import streamlit as st
from PIL import Image

st.set_page_config(layout="wide")


def add_logo(logo_path, width, height):
    """Read and return a resized logo"""
    logo = Image.open(logo_path)
    modified_logo = logo.resize((width, height))
    return modified_logo

my_logo = add_logo(logo_path="Soosthsayer_logo.png", width=280, height=100)
st.sidebar.image(my_logo)

st.title("References")
st.markdown("<hr/>", unsafe_allow_html=True) 
st.markdown(
            """
                
            - Korunovic et. al., "In Silico Optimization of Femoral Fixator Position and Configuration by Parametric CAD Model", Materials 2019, 12, 2326; doi:10.3390/ma12142326..
            - https://www.kaggle.com/code/milanzdravkovic/design-time-product-structural-analysis-assistance/notebook.
            """
            )
    