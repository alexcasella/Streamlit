
# -*- coding: utf-8 -*-
"""
Created on Thu May  2021
@author: Alex https://github.com/alexcasella
"""
# Awesome Streamlit
import streamlit as st

# Add pages -- see those files for deatils within
from page_joseph import page_joseph
from page_rohit import page_rohit
from page_introduction import page_introduction

# Use random seed
import numpy as np
np.random.seed(1)


# Set the default elements on the sidebar
st.set_page_config(page_title='SteamingHot',layout='wide')

st.markdown("<h1 style='text-align: right; color: grey;'> \
                Steaming Hot </h1>", unsafe_allow_html=True)

st.sidebar.write(" ")


def main():
    """
    Register pages to Explore and Fit:
        page_introduction - contains page with images and brief explanations
        page_joseph - contains joseph's algorithm
        page_rohit - rohit's algorithm
    """

    pages = {
        "Introduction": page_introduction,
        "Time Prediction": page_joseph,
        "Feature Score": page_rohit,
    }

    st.sidebar.title("Main options")

    # Radio buttons to select desired option
    page = st.sidebar.radio("Select:", tuple(pages.keys()))
                                
    # Display the selected page with the session state
    pages[page]()

    # Write About
    st.sidebar.header("About")
    st.sidebar.warning(
            """
            SteamingHot app is created and maintained by 
            **Alex Casella**. If you like this app please star its
            [**GitHub**](https://github.com/alexcasella)
            repo, share it and feel free to open an issue if you find a bug 
            or if you want some additional features.
            """
    )


if __name__ == "__main__":
    main()
