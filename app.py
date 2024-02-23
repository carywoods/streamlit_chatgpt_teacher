import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder


# Load your data
df = pd.read_csv('ChatGPT_Teacher_sheet1.csv')

# Title of your app
st.title('EDA of Dataset')

# Show the raw data
if st.checkbox('Show raw data'):
    st.write(df)
