import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load your data
df = pd.read_csv('path/to/your/data.csv')

# Title of your app
st.title('EDA of Dataset')

# Show the raw data
if st.checkbox('Show raw data'):
    st.write(df)

# Basic stats
if st.checkbox('Show basic statistics'):
    st.write(df.describe())

# Visualizations
st.sidebar.title('Visualizations')
plot_type = st.sidebar.selectbox('Choose the type of plot', ['Histogram', 'Boxplot', 'Correlation Heatmap'])

if plot_type == 'Histogram':
    column = st.sidebar.selectbox('Choose the column to plot histogram', df.columns)
    bins = st.sidebar.slider('Number of bins', min_value=10, max_value=100, value=30)
    plt.hist(df[column].dropna(), bins=bins)
    plt.title(f'Histogram of {column}')
    st.pyplot(plt)

elif plot_type == 'Boxplot':
    column = st.sidebar.selectbox('Choose the column for boxplot', df.columns)
    sns.boxplot(y=df[column])
    plt.title(f'Boxplot of {column}')
    st.pyplot(plt)

elif plot_type == 'Correlation Heatmap':
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    st.pyplot(plt)
