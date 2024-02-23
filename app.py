import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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


# Load your dataset
df1 = df

# Assuming 'target' is the name of your target variable
# and the rest of the columns are features
X = df1.drop('NUMBER OF LIKES', axis=1)
y = df1['target']

# If your target variable is categorical, encode it
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

# Streamlit app title
st.write('Your Dataset Classification')

# Model training
model = KNeighborsClassifier(n_neighbors=3)  # You might want to choose a model that suits your data
model.fit(X_train, y_train)

# Making predictions
predictions = model.predict(X_test)

# Displaying results
st.write("Classification Report:")
st.text(classification_report(y_test, predictions, target_names=le.classes_))

st.write("Confusion Matrix:")
st.write(confusion_matrix(y_test, predictions))

# User input for predictions
# You'll need to create input fields for each feature
# For example:
# feature1 = st.number_input('Feature 1')
# Adjust this part based on your actual features

# Button for prediction
if st.button('Predict'):
    # Adjust this to match the input format of your model
    user_prediction = model.predict([[feature1, feature2, ...]])  # Replace feature1, feature2, ... with actual features
    st.write(f"The predicted class is: {le.inverse_transform(user_prediction)[0]}")

