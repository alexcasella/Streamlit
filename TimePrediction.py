# Import the relevant packages

import streamlit as st

import pandas as pd
import numpy as np

# Import the prediction function

from Joseph_alg import ult_pred
from Joseph_alg import get_values


# Import the dataframe

data = pd.read_csv('data18m.csv', index_col=0)
smooth = pd.read_csv('Smooth_data18m_pure.csv', index_col=0)
nonsmooth = pd.read_csv('nonsmooth_data18m_pure.csv', index_col=0)

smooth_data = get_values(smooth)
nonsmooth_data = get_values(nonsmooth)


# Write the headline

st.write("""

# Time Prediction Algorithm

Enter Brief Description Here

""")


## Set up the sidebar

# Form containing the inputs

form = st.sidebar.form('my_form')

# Header

form.header('User Input Values')

No_Of_Months = form.slider('Select No Of Months To Predict',1,12,6)

# Initalize Game_Num for the if statement

Game_Num = 0

Game_Num = int(form.text_input('Enter A Game Number From The List Below',3597,max_chars=4,help = 'Enter a number between 0 and ' + str(len(data)-1)))

Game_Name = data.iloc[Game_Num].Name
    
data_tmp = {'Game Number': Game_Num,
            'Game': Game_Name,
            'Number of Months': No_Of_Months}
df = pd.DataFrame(data_tmp, index=[0])

data_1 = pd.DataFrame(data,columns = ['Name']).sort_values('Name')

form.write(data_1.style.set_properties(**{'text-align': 'left'}))

# Now add a submit button to the form:
form.form_submit_button("Submit")

## Set up the output

# Print the choice made

st.subheader('User\'s choice:')

st.write(df)

# ### Imaginary Examples:

# In[188]:

name = df['Game'].values[0]
months = df['Number of Months'].values[0]

game = nonsmooth_data[data.loc[data['Name']==name].index[0]][:12]
real_data = nonsmooth_data[data.loc[data['Name']==name].index[0]][11:12+months]

subdata = nonsmooth_data.copy()
del subdata[data.loc[data['Name']==name].index[0]]
smooth_subdata = smooth_data.copy()
del smooth_subdata[data.loc[data['Name']==name].index[0]]

# Print the closest three games

[pred, close_index,close_games] = ult_pred(game, train = subdata, smooth_train = smooth_subdata, real_data = real_data, horizon = months)
