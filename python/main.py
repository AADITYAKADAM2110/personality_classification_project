from data import load_data
from evaluation import evaluate_models
from preprocess import preprocess_data
from models import train_all_models
import pandas as pd

def main():
    # Load data
    data = load_data()
    
    # Preprocess data
    X, y, X_train_scaled, X_test_scaled, y_train, y_test = preprocess_data()

    # Train models
    trained_models = train_all_models()

    # Evaluate models
    evaluate_models()

if __name__ == "__main__":
    main()