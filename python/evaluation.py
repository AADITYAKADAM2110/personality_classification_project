from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from models import train_all_models
from preprocess import preprocess_data


def evaluate_models():    

    trained_models = train_all_models()

    for model_name, model in trained_models.items():

        X, y, X_train_scaled, X_test_scaled, y_train, y_test = preprocess_data()
        y_pred = model.predict(X_test_scaled)
        print(f"{model_name} Accuracy: {model.score(X_test_scaled, y_test)}")
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))

        # Predict probabilities
        y_probs = model.predict_proba(X_test_scaled)[:, 1]

        # ROC AUC Score
        auc_score = roc_auc_score(y_test, y_probs)
        print(f"ROC AUC Score for {model_name}: {auc_score}")

        # Plot ROC Curve
        fpr, tpr, thresholds = roc_curve(y_test, y_probs)
        plt.plot(fpr, tpr, label =f"{model_name} (AUC = {auc_score:.2f})")
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for {model_name}')
        plt.legend()
        plt.show()
