# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 14:49:11 2022

@author: susym
"""
import streamlit as st
from PIL import Image


st.set_page_config(layout="wide")

#st.set_page_config(page_title="AI Model Training",layout = "wide")

#st.title('AI based assistance for design engineers to accelerate the product development process')

new_title = '<p style="font-family:sans-serif;text-align: center; color:#87CEEB; font-size: 42px;">AI based assistance for design engineers to accelerate the product development process</p>'
st.markdown(new_title, unsafe_allow_html=True)

def add_logo(logo_path, width, height):
    """Read and return a resized logo"""
    logo = Image.open(logo_path)
    modified_logo = logo.resize((width, height))
    return modified_logo

my_logo = add_logo(logo_path="Soosthsayer_logo.png", width=280, height=100)
st.sidebar.image(my_logo)

    


#st.markdown("<h8 style='text-align: center; color: black;'>A no-code AI platform which learns from historical simulation/test data and predict the performance of a future design.</h8>", unsafe_allow_html=True)
#st.markdown("*A no-code AI platform which learns from historical simulation/test data and predict the performance of a future design.*")
#subtitle = '<p style="font-family:sans-serif;text-align: center; color:black; font-size: 42px;">

colT1,colT2,colT3 = st.columns([1,2.5,1])
with colT2:
    st.markdown("<h8 style='text-align: center; color: black;'>A no-code AI platform which learns from historical simulation/test data and predict the performance of a future design.</h8>", unsafe_allow_html=True)
