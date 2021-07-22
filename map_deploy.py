import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

px.set_mapbox_access_token("pk.eyJ1IjoiZmxhdmlvbG9zcyIsImEiOiJja3Fjamx6ZnEwNXcwMndqbzVsYnU3a29pIn0.3T-wJQDphQBJO148DKmf2Q")


def create_map(df_raw, slicer, type):
    df = df_raw[(df_raw['population'] >= slicer[0]) & (df_raw['population'] <= slicer[1])]

    size = 'amount_tsh' if type == 'train' or type == 'test' else None
    amount = 'Amount of Water Available' if type == 'train' or type == 'test' else None
    title = 'Tanzania WaterPumps' if type == 'train' or type == 'test' else 'Tanzania WaterPumps (Zero amount of water left)'
    status = 'Status(Predicted)' if type == 'test' else 'Status '

    if len(df) != 0:
        fig = px.scatter_mapbox(df, lat='latitude', lon='longitude', color='status_group', size=size,
                            color_discrete_sequence=['steelblue', '#F2722A', 'cyan'], size_max=25, zoom=4,
                            hover_name='wpt_name', hover_data=['population', 'amount_tsh'], title=title,
                            labels={size: amount, 'population': 'Population ', 'status_group': status})

        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig)

    else:
        fig = px.scatter_mapbox(df_raw, lat='latitude', lon='longitude', zoom=4)
        fig.update_traces(marker={'size':0}, hoverinfo='skip', hovertemplate=None)

        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig)


train_labels = pd.read_csv('train-labels.csv')
train_values = pd.read_csv('train-values.csv')
df_train = pd.merge(train_labels, train_values, on='id')
df_train.status_group = df_train.status_group.replace({'functional': 'Functional',
                                                       'functional needs repair': 'Needs repair',
                                                       'non functional': 'Non functional'})

test_values = pd.read_csv('test-values.csv')
test_preds = pd.read_csv('submission_catboost_v3.csv')
df_test = pd.merge(test_values, test_preds, on='id')
df_test.status_group = df_test.status_group.replace({'functional': 'Functional',
                                                     'functional needs repair': 'Needs repair',
                                                     'non functional': 'Non functional'})

df = pd.concat([df_train, df_test]).set_index('id')
df_zeroes = df[df['amount_tsh'] == 0]

df_train = df_train[df_train['longitude'] > 10]
df_train = df_train[df_train['amount_tsh'] <= 70000]
df_train = df_train[df_train['amount_tsh'] != 0]

df_test = df_test[df_test['longitude'] > 10]
df_test = df_test[df_test['amount_tsh'] <= 70000]
df_test = df_test[df_test['amount_tsh'] != 0]

st.title('Tanzania Water Pumps - A Interactive Map Visualization')
st.sidebar.image('AJUDADOS_LOGO.png', width=200)
st.markdown('This is a map visualization of the Tanzania Water Pump challenge, \
             using data from Taarifa and the Tanzanian Ministry of Water')

map_radio = st.sidebar.radio('Map: ', 
                            ('Observed and Labeled Pumps', 'Predicted Pumps', 'Pumps with zero water left'))
# train_button = st.sidebar.button('Observed and Labeled Pumps')
# test_button = st.sidebar.button('Predicted Pumps')
# zeroes_button = st.sidebar.button('Pumps with zero water left')

pop_slicer = st.sidebar.slider('Population around the well', value=[int(min(df['population'].values)), 
                                                                    int(max(df['population'].values))])

if map_radio == 'Observed and Labeled Pumps':
    create_map(df_train, pop_slicer, type='train')

elif map_radio == 'Predicted Pumps':
    create_map(df_train, pop_slicer, type='test')

elif map_radio == 'Pumps with zero water left':
    create_map(df_train, pop_slicer, type='zeros')
