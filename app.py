import streamlit as st
import pandas as pd
from textblob import TextBlob

# Ensure TextBlob corpora is downloaded
import nltk
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
    # Adjust the path if your dataset is located elsewhere
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

st.write(filtered_df)
