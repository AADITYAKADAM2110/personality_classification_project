import pandas as pd
from data import load_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data():

    data = load_data()


    data['Stage_fear'] = data['Stage_fear'].str.replace('Yes', '1').str.replace('No', '0').astype(int)
    data['Drained_after_socializing'] = data['Drained_after_socializing'].str.replace('Yes', '1').str.replace('No', '0').astype(int)
    data['Personality'] = data['Personality'].str.replace('Introvert', '0').str.replace('Extrovert', '1').astype(int)

    X = data.drop('Personality', axis=1)
    y = data['Personality']

    scaler = StandardScaler()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X, y, X_train_scaled, X_test_scaled, y_train, y_test