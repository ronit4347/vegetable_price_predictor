# model_trainer.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import joblib
import os

print("Starting model training script...")

# --- 1. Generate Synthetic Data ---
def generate_synthetic_data(num_samples=1000):
    """
    Generates synthetic data for vegetable prices.
    Features: Date, Vegetable Type
    Target: Price
    """
    print("Generating synthetic data...")
    dates = pd.to_datetime(pd.date_range(start='2023-01-01', periods=num_samples, freq='D'))
    vegetables = ['Tomato', 'Potato', 'Onion', 'Carrot', 'Spinach', 'Cabbage', 'Bell Pepper', 'Cucumber']
    data = []

    for date in dates:
        for veg in vegetables:
            # Base price for each vegetable
            base_price = {
                'Tomato': 30, 'Potato': 20, 'Onion': 25, 'Carrot': 35,
                'Spinach': 15, 'Cabbage': 20, 'Bell Pepper': 50, 'Cucumber': 25
            }[veg]

            # Add seasonality (e.g., higher prices in certain months)
            month = date.month
            season_factor = 1.0
            if veg in ['Tomato', 'Onion']: # Generally higher in off-season or due to supply issues
                if month in [6, 7, 8, 9]: # Monsoon months, often higher prices
                    season_factor = 1.3
                elif month in [1, 2, 11, 12]: # Winter, good harvest, lower prices
                    season_factor = 0.8
            elif veg in ['Spinach', 'Cabbage']: # Winter vegetables
                if month in [6, 7, 8]:
                    season_factor = 1.5
                elif month in [11, 12, 1, 2]:
                    season_factor = 0.7

            # Add random noise
            noise = np.random.uniform(-5, 5)

            price = (base_price * season_factor) + noise
            price = max(10, round(price, 2)) # Ensure price is not too low

            data.append([date, veg, price])

    df = pd.DataFrame(data, columns=['Date', 'Vegetable', 'Price'])
    print(f"Generated {len(df)} data points.")
    return df

# --- 2. Preprocess Data ---
def preprocess_data(df):
    """
    Preprocesses the DataFrame for model training.
    Encodes categorical features and extracts numerical date features.
    """
    print("Preprocessing data...")
    # Extract numerical features from Date
    df['day_of_year'] = df['Date'].dt.dayofyear
    df['month'] = df['Date'].dt.month
    df['year'] = df['Date'].dt.year # Add year as a feature

    # Encode 'Vegetable' column
    le = LabelEncoder()
    df['vegetable_encoded'] = le.fit_transform(df['Vegetable'])
    print("Data preprocessing complete.")
    return df, le

# --- 3. Train Model ---
def train_model(df):
    """
    Trains a Linear Regression model on the preprocessed data.
    """
    print("Training model...")
    features = ['day_of_year', 'month', 'year', 'vegetable_encoded']
    target = 'Price'

    X = df[features]
    y = df[target]

    # Split data into training and testing sets (optional for this simple example, but good practice)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    print("Model training complete.")
    return model

# --- Main execution ---
if __name__ == "__main__":
    # Ensure the 'models' directory exists
    models_dir = 'models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"Created directory: {models_dir}")

    # Generate data
    df = generate_synthetic_data()

    # Preprocess data and get label encoder
    df_processed, label_encoder = preprocess_data(df)

    # Train model
    model = train_model(df_processed)

    # Save the trained model and label encoder
    model_path = os.path.join(models_dir, 'linear_regression_model.pkl')
    encoder_path = os.path.join(models_dir, 'label_encoder.pkl')

    joblib.dump(model, model_path)
    joblib.dump(label_encoder, encoder_path)

    print(f"Model saved to: {model_path}")
    print(f"Label Encoder saved to: {encoder_path}")
    print("Model training script finished successfully.")