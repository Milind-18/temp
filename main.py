from pipeline.train import train_model, evaluate_model, save_model
from pipeline.preprocess import load_data, preprocess_data

def main():
    # Load and preprocess the data
    df = load_data(path="data/iris.csv")  # Adjust path if needed
    X_train, X_test, y_train, y_test = preprocess_data(df)

    # Train the model
    model = train_model(X_train, y_train)

    # Evaluate the model
    evaluate_model(model, X_test, y_test)

    # Save the trained model
    save_model(model, path="random_forest_model.joblib")

if __name__ == "__main__":
    main()
