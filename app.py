import pandas as pd
import streamlit as st
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

# Ensure TextBlob corpora is downloaded
nltk.download('punkt')

# Sentiment and likes categorization functions remain unchanged

# Load the dataset function remains unchanged

# Load data
df = load_data()

# Streamlit app structure
st.title('Sentiment and Classification Analysis based on Description and Likes')

# Data filtering logic remains unchanged

# New Visualization: Grouped bar chart for Sentiment vs Likes Category
st.subheader('Sentiment vs Likes Category Distribution')

# Creating a pivot table for the counts of likes categories within each sentiment
pivot_df = df.groupby(['Sentiment', 'Likes_Category']).size().unstack(fill_value=0)
pivot_df.plot(kind='bar', stacked=False, figsize=(10, 7))

plt.ylabel('Count')
plt.title('Distribution of Likes Categories within Each Sentiment')
st.pyplot(plt)
