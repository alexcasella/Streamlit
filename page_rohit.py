# Import the relevant packages

import streamlit as st

import pandas as pd
import numpy as np

# Import the prediction function

from Feature_Importance import plot_scores

# Import the relevant data

data = pd.read_csv('AllData.csv').set_index('Name')
data=data[data['mean']>1]
proc_data=data.drop(['Metacritic','App_id','ReleaseDate','SteamSpyOwners', 'SteamSpyOwnersVariance',
                      'SteamSpyPlayersEstimate', 'SteamSpyPlayersVariance', 'PriceCurrency','SupportEmail', 
                      'SupportURL', 'AboutText','Background', 'ShortDescrip', 'DetailedDescrip', 'DRMNotice',
                      'ExtUserAcctNotice', 'HeaderImage', 'LegalNotice', 'Reviews','RecommendationCount',
                      'SupportedLanguages', 'Website', 'PCMinReqsText', 'PCRecReqsText','SteamSpyOwnersNew', 
                      'peak_max','peak_mean', 'max','LinuxMinReqsText', 'LinuxRecReqsText', 'MacMinReqsText',
                      'MacRecReqsText','RequiredAge','DemoCount','DLCCount','DeveloperCount','MovieCount',
                      'PackageCount','PublisherCount','ScreenshotCount','AchievementCount',
                      'AchievementHighlightedCount','PriceInitial','PriceFinal','mean'],axis=1)

# Import the dataframe
def page_rohit():
#    data = pd.read_csv('data18m.csv', index_col=0)

    # Write the headline

    st.markdown("<h2 style='text-align: center;'> Welcome To </h2>", 
                    unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center;'> The Feature Score Algorithm </h1>", 
                    unsafe_allow_html=True)

    # Description of the algorithm

    st.markdown("""
                This algorithm allows you to compare the relevance of a set of features in a chosen popularity class. \n
                The popularity class of a game is:
                - 1: _if the average number of players per day is less than 10_
                - 2: _if the average number of players per day is between 10 and 100_
                - 3: _if the average number of players per day is between 100 and 1000_
                - 4: _if the average number of players per day is between 1000 and 10000_
                - 5: _if the average number of players per day is more than 10000_ \n
                The relevance of a feature F in a given class C is determined by its:
                #### - Importance Score:
                _This is the number of games in class C with feature F, over the total number of games in class C_.
                #### - Relative Importance Score:
                _This is the importance score of feature F in class C, divided by the sum over all classes of all importance scores of feature F in each class._
                #### - Benefit Score:
                _This is the number of games in class C with feature F, over the total number of games with feature F._
                #### - Relative Benefit and Importance Score:
                _This is the importance score of feature F in class C, multiplied by the benefit score of feature F in class C._
                """)

    st.info("""
                To run the algorithm: \n
                - Choose a popularity class (1-5)
                - Select the set of features you want to compare
                - Press Submit! \n
                The algorithm returns four plots, one for each score, with the selected features.
                """)

    ## Set up the sidebar

    # Form containing the inputs

    form = st.sidebar.form('my_form')

    # Header

    form.header('User Input Values')

    grade_choice = form.slider('Select The Popularity Class',1,5,5)

    # Initalize Game_Num for the if statement

    all_features = list(proc_data.columns)
    
    feature = ['']*len(all_features)

    features = []

    feature[0] = form.checkbox(all_features[0],value=True)
    
    for i in range(1,len(all_features)):
        feature[i] = form.checkbox(all_features[i])

    for i in range(0,len(all_features)):
        if feature[i]:
            features.append(all_features[i])
    
    # Now add a submit button to the form:
    form.form_submit_button("Submit")

    ## Set up the output

    # Print the choice made

    st.subheader('Selected Features:')

    st.write(features)

    # ### Imaginary Examples:

    # In[188]:

    plot_scores(feats=features,grade=grade_choice)
    
    # Print the closest three games

    return
