# predictor.py
import joblib
import pandas as pd
import os
from datetime import datetime

# Define the path to the models directory
MODELS_DIR = 'models'
MODEL_PATH = os.path.join(MODELS_DIR, 'linear_regression_model.pkl')
ENCODER_PATH = os.path.join(MODELS_DIR, 'label_encoder.pkl')

# Global variables to store loaded model and encoder
_model = None
_label_encoder = None

def _load_model_and_encoder():
    """
    Loads the trained model and label encoder.
    This function is designed to be called once to avoid
    reloading on every prediction.
    """
    global _model, _label_encoder
    if _model is None or _label_encoder is None:
        try:
            print(f"Loading model from: {MODEL_PATH}")
            _model = joblib.load(MODEL_PATH)
            print(f"Loading label encoder from: {ENCODER_PATH}")
            _label_encoder = joblib.load(ENCODER_PATH)
            print("Model and encoder loaded successfully.")
        except FileNotFoundError:
            print(f"Error: Model or encoder file not found. Please ensure '{MODELS_DIR}' directory exists and contains 'linear_regression_model.pkl' and 'label_encoder.pkl'.")
            print("You might need to run 'python model_trainer.py' first.")
            _model = None
            _label_encoder = None
        except Exception as e:
            print(f"An error occurred while loading model/encoder: {e}")
            _model = None
            _label_encoder = None

def get_available_vegetables():
    """
    Returns a list of vegetables the model was trained on.
    Requires the label encoder to be loaded.
    """
    _load_model_and_encoder() # Ensure encoder is loaded
    if _label_encoder:
        return list(_label_encoder.classes_)
    return []

def predict_price(date_str: str, vegetable_type: str) -> float:
    """
    Predicts the price of a vegetable for a given date.

    Args:
        date_str (str): Date in 'YYYY-MM-DD' format.
        vegetable_type (str): Name of the vegetable (e.g., 'Tomato').

    Returns:
        float: Predicted price. Returns -1 if prediction fails.
    """
    _load_model_and_encoder() # Ensure model and encoder are loaded

    if _model is None or _label_encoder is None:
        return -1.0 # Indicate failure to predict due to missing model/encoder

    try:
        # Convert date string to datetime object
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')

        # Extract numerical features
        day_of_year = date_obj.timetuple().tm_yday
        month = date_obj.month
        year = date_obj.year

        # Encode vegetable type
        try:
            vegetable_encoded = _label_encoder.transform([vegetable_type])[0]
        except ValueError:
            print(f"Error: '{vegetable_type}' is not a recognized vegetable type by the model.")
            print(f"Available vegetables: {list(_label_encoder.classes_)}")
            return -1.0 # Indicate failure due to unknown vegetable

        # Create a DataFrame for prediction (model expects 2D array-like input)
        input_data = pd.DataFrame([[day_of_year, month, year, vegetable_encoded]],
                                  columns=['day_of_year', 'month', 'year', 'vegetable_encoded'])

        # Make prediction
        predicted_price = _model.predict(input_data)[0]

        # Ensure price is non-negative and round it
        return max(0.0, round(predicted_price, 2))

    except ValueError as ve:
        print(f"Input error: {ve}. Please check date format (YYYY-MM-DD).")
        return -1.0
    except Exception as e:
        print(f"An unexpected error occurred during prediction: {e}")
        return -1.0

# Example of how to use this module (for testing purposes)
if __name__ == "__main__":
    # Make sure to run model_trainer.py first to create the model files
    # python model_trainer.py

    print("\n--- Testing predictor.py ---")
    available_veg = get_available_vegetables()
    if available_veg:
        print(f"Model trained on: {available_veg}")

        test_date = "2024-07-01"
        test_veg = "Tomato"
        price = predict_price(test_date, test_veg)
        if price != -1.0:
            print(f"Predicted price for {test_veg} on {test_date}: ₹{price}")
        else:
            print(f"Could not predict price for {test_veg} on {test_date}.")

        test_date_2 = "2024-01-15"
        test_veg_2 = "Potato"
        price_2 = predict_price(test_date_2, test_veg_2)
        if price_2 != -1.0:
            print(f"Predicted price for {test_veg_2} on {test_date_2}: ₹{price_2}")
        else:
            print(f"Could not predict price for {test_veg_2} on {test_date_2}.")

        test_date_3 = "2024-08-20"
        test_veg_3 = "Spinach"
        price_3 = predict_price(test_date_3, test_veg_3)
        if price_3 != -1.0:
            print(f"Predicted price for {test_veg_3} on {test_date_3}: ₹{price_3}")
        else:
            print(f"Could not predict price for {test_veg_3} on {test_date_3}.")

        test_date_4 = "2024-03-10"
        test_veg_4 = "NonExistentVeg" # Test with an unknown vegetable
        price_4 = predict_price(test_date_4, test_veg_4)
        if price_4 != -1.0:
            print(f"Predicted price for {test_veg_4} on {test_date_4}: ₹{price_4}")
        else:
            print(f"Could not predict price for {test_veg_4} on {test_date_4}.")
    else:
        print("Cannot test predictor.py as model and encoder could not be loaded.")
