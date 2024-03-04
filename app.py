import streamlit as st
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

# Ensure TextBlob corpora is downloaded
nltk.download('punkt')

# Function to calculate sentiment
def calculate_sentiment(text):
    try:
        sentiment = TextBlob(text).sentiment.polarity
        return 'Positive' if sentiment > 0 else 'Negative' if sentiment < 0 else 'Neutral'
    except:
        return 'Neutral'

# Function to categorize likes
def categorize_likes(likes):
    try:
        if likes <= 100:
            return 'Low'
        elif likes <= 500:
            return 'Medium'
        else:
            return 'High'
    except:
        return 'Unknown'

# Load the dataset
@st.cache
def load_data():
    df = pd.read_csv('chat_teacher_fb.csv')
    df['Sentiment'] = df['description'].apply(calculate_sentiment)
    df['Likes_Category'] = df['likes'].apply(categorize_likes)
    return df

# Load data
df = load_data()

# Streamlit app structure
st.title('Sentiment and Classification Analysis based on Description and Likes')

# Displaying the dataframe
if st.checkbox('Show raw data'):
    st.write(df)

# Filters for sentiment and likes category
sentiment_filter = st.sidebar.selectbox('Select Sentiment', options=['All', 'Positive', 'Neutral', 'Negative'])
likes_filter = st.sidebar.selectbox('Select Likes Category', options=['All', 'Low', 'Medium', 'High'])

# Filtering data based on selection
filtered_df = df
if sentiment_filter != 'All':
    filtered_df = filtered_df[filtered_df['Sentiment'] == sentiment_filter]
if likes_filter != 'All':
    filtered_df = filtered_df[filtered_df['Likes_Category'] == likes_filter]

# Visualization: Grouped bar chart for Sentiment vs Likes Category Distribution
st.subheader('Sentiment vs Likes Category Distribution')

# Creating a pivot table for the counts of likes categories within each sentiment
pivot_df = filtered_df.groupby(['Sentiment', 'Likes_Category']).size().reset_index(name='Count')
pivot_df = pivot_df.pivot_table(index='Sentiment', columns='Likes_Category', values='Count', fill_value=0)

# Plotting
fig, ax = plt.subplots()
pivot_df.plot(kind='bar', stacked=False, ax=ax)
ax.set_ylabel('Count')
ax.set_title('Distribution of Likes Categories within Each Sentiment')
st.pyplot(fig)

st.write(filtered_df)
