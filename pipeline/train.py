from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

def train_model(X_train, y_train):
    model = RandomForestClassifier()
    model.fit(X_train,y_train)
    return model

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc}")
    return acc

def save_model(model, path="model.joblib"):
    joblib.dump(model, path)



