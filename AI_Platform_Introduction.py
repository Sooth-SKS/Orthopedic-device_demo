# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 14:49:11 2022

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

st.markdown("<h1 style='text-align: center; color: black'>AI based assistance for design engineers to accelerate the product development process</h1>", unsafe_allow_html=True)
st.write("Don’t waste your past design experiences. Learn from them using highly user-friendly AI assistance tool and be more creative and confident in your next design.")
st.markdown("<hr/>", unsafe_allow_html=True)

st.markdown(
"""
Properties of AI assistance tool :    
- *Say no to your local desktop. Host all your data on cloud*.  
- *Highly user-friendly to engineers. They don’t need to know coding or AI*.
- *Highly customized for the problems you solve*.
- *Low cost to initiate and to run the platform*
"""
)
 
