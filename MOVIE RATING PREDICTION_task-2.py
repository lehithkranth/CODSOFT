import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error

# Load dataset
df = pd.read_csv('movie_dataset.csv', encoding='latin1')

# Convert Duration to numeric (remove 'min' and convert to int)
df['Duration'] = df['Duration'].str.replace(' min', '', regex=True).astype(float)

# Combine actors into a single column
df['actors'] = df[['Actor 1', 'Actor 2', 'Actor 3']].fillna('').agg(', '.join, axis=1)

# Select features and target
X = df[['Genre', 'Director', 'actors', 'Duration']]
y = df['Rating'].fillna(df['Rating'].mean())  # Fill missing ratings with mean

# Handle missing values in numeric column
num_imputer = SimpleImputer(strategy='mean')
X['Duration'] = num_imputer.fit_transform(X[['Duration']])

# Define preprocessing steps
categorical_features = ['Genre', 'Director']
text_features = ['actors']
numeric_features = ['Duration']

categorical_transformer = OneHotEncoder(handle_unknown='ignore')
text_transformer = TfidfVectorizer(max_features=500)  # Limit to 500 features for speed
numeric_transformer = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features),
        ('text', text_transformer, 'actors'),
        ('num', numeric_transformer, numeric_features)
    ]
)

# Choose a model (RandomForestRegressor or LinearRegression for faster training)
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=42))  # Faster training
    # ('regressor', LinearRegression())  # Use this for an even faster model
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
print("Training model...")
model.fit(X_train, y_train)
print("Model training completed!")

# Predict
y_pred = model.predict(X_test)

# Evaluate performance
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse:.4f}')
