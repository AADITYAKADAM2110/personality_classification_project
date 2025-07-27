import pandas as pd


def load_data():
    data = pd.read_csv(r'C:\Users\DELL\Desktop\Classification_Project\dataset\personality_datasert.csv')

    print("Data loaded successfully.")

    return data
    
load_data()    