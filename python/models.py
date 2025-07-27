from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from preprocess import preprocess_data

def train_all_models():
    X, y, X_train_scaled, X_test_scaled, y_train, y_test = preprocess_data()

    models = {
        "Logistic Regression": LogisticRegression(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "SVC": SVC(probability=True)
    }


    trained_models = {}

    for model_name, model in models.items():
        model.fit(X_train_scaled, y_train)
        trained_models[model_name] = model
        print(f"{model_name} trained successfully.")
    return trained_models