# Import the relevant packages

import streamlit as st

import pandas as pd
import numpy as np

# Import the prediction function

from predict_owners import predict_owners


# Import the dataframe
def page_kanishk():


    # Write the headline

    st.markdown("<h2 style='text-align: center;'> Welcome To </h2>", 
                    unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center;'> The Owners Prediction Algorithm </h1>", 
                    unsafe_allow_html=True)

    # Description of the algorithm

    st.markdown("""
                This algorithm allows you to predict the number of owners of a game having the following features in its store page:
                - PackageCount: _the number of package deals the game is part of_
                - MovieCount: _the number of videos about the game_
                - ScreenshoCount: _the number of screenshots of the game_
                - AchievementCount: _the number of achievements available for the players_
                - PriceMean: _the average between the inital and final price_ \n
                The output is a monthly sale prediction of the game.
                """)

    st.info("""
                To run the algorithm: \n
                - Choose the PackageCount (1-18)
                - Select the MovieCount (0-20)
                - Select the ScreenshotCount (0-180)
                - Select the AchievementCount (0-9821)
                - Insert the PriceMean (0.5-449.99)
                - Press Submit! \n
                The algorithm returns a plot with a monthly sale prediction of the game.
                """)


    ## Set up the sidebar

    # Form containing the inputs

    form = st.sidebar.form('my_form')

    # Header

    form.header('User Input Values')

    packagecount = form.slider('Select the PackageCount',1,18,1)
    moviecount = form.slider('Select the MovieCount',0,20,3)
    screenshotcount = int(form.text_input('Select the ScreenshotCount',5,max_chars=3,help = 'Enter a number between 0 and 180'))
    achievementcount = int(form.text_input('Select the AchievementCount',10,max_chars=4,help = 'Enter a number between 0 and 9821'))
    pricemean = float(form.text_input('Select the PriceMean',9.99,max_chars=6,help = 'Enter a number between 0.5 and 449.99'))

    # Initalize Game_Num for the if statement

    data_tmp = {'PackageCount': packagecount,
                'MovieCount': moviecount,
                'ScreenshotCount': screenshotcount,
                'AchievementCount': achievementcount,
                'PriceMean': pricemean
                }
    df = pd.DataFrame(data_tmp, index=[0])

    # Now add a submit button to the form:
    form.form_submit_button("Submit")

    ## Set up the output

    # Print the choice made

    st.subheader('User\'s choice:')

    st.write(df)


    # Print the closest three games

    predict_owners(packagecount,moviecount,screenshotcount,achievementcount,pricemean)

    return
