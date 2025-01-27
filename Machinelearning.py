import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load dataset
df = pd.read_csv('netflix_titles.csv')
print(df.head())
# Clean the data - Handling missing values and non-numeric columns
df['release_year'] = pd.to_numeric(df['release_year'], errors='coerce')  # Ensure release_year is numeric
df.dropna(subset=['release_year'], inplace=True)  # Drop rows with missing release_year

# Handling categorical columns using LabelEncoder
label_encoder = LabelEncoder()

# Fill missing values in categorical columns
df['director'] = df['director'].fillna('Unknown')  # Fill missing values with a placeholder
df['cast'] = df['cast'].fillna('Unknown')  # Fill missing values with a placeholder
df['country'] = df['country'].fillna('Unknown')  # Fill missing values with a placeholder
df['rating'] = df['rating'].fillna('Unknown')  # Fill missing values with a placeholder

# Fit the LabelEncoder to the entire dataset to ensure it can handle unseen values
df['director'] = label_encoder.fit_transform(df['director'].astype(str))  # Convert director to numeric
df['cast'] = label_encoder.fit_transform(df['cast'].astype(str))  # Convert cast to numeric
df['country'] = label_encoder.fit_transform(df['country'].astype(str))  # Convert country to numeric
df['rating'] = label_encoder.fit_transform(df['rating'].astype(str))  # Convert rating to numeric
df['duration'] = df['duration'].apply(lambda x: int(x.split(' ')[0]) if isinstance(x, str) and 'min' in x else 0)  # Extract numeric part of duration

# Features and target
X = df[['director', 'cast', 'country', 'rating', 'duration']]  # Features
y = df['release_year']  # Target

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline for preprocessing and model training
pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values
    ('scaler', StandardScaler()),  # Scale features
    ('regressor', LinearRegression())  # Use Linear Regression model
])

# Train the model
pipeline.fit(X_train, y_train)

# Model evaluation
train_score = pipeline.score(X_train, y_train)
test_score = pipeline.score(X_test, y_test)

# Predictions
predictions = pipeline.predict(X_test)

# Output results
print(f"Train Score: {train_score}")
print(f"Test Score: {test_score}")
print(f"Predictions: {predictions[:5]}")  # Show first 5 predictions

# Handle unseen labels for new data:
def handle_unseen_labels(new_data, label_encoder):
    transformed_data = []
    for value in new_data:
        try:
            transformed_value = label_encoder.transform([value])[0]  # Try transforming
        except ValueError:  # If unseen label, map it to a default value
            transformed_value = label_encoder.transform(['Unknown'])[0]  # Use the encoded value for 'Unknown'
        transformed_data.append(transformed_value)
    return transformed_data

# Example of predicting for new data (with unseen labels)
new_data = [
    'Steven Spielberg',  # director (may be unseen)
    'Tom Hanks',  # cast (may be unseen)
    'United States',  # country (may be unseen)
    7.8,  # rating (numeric already)
    120  # duration (numeric already)
]

# Convert new data to a pandas DataFrame to maintain feature names
new_data_transformed = [
    handle_unseen_labels([new_data[0]], label_encoder)[0], 
    handle_unseen_labels([new_data[1]], label_encoder)[0], 
    handle_unseen_labels([new_data[2]], label_encoder)[0], 
    new_data[3], 
    new_data[4]
]

# Convert new data to DataFrame with column names
new_data_df = pd.DataFrame([new_data_transformed], columns=['director', 'cast', 'country', 'rating', 'duration'])

# Make prediction for the new data
new_prediction = pipeline.predict(new_data_df)
print(f"Prediction for new data: {new_prediction}")
