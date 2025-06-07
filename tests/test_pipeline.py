from pipeline import preprocess, train
import pandas as pd

def test_loaddata():
    df = preprocess.load_data("data/iris.csv")
    assert not df.empty

def test_preprocess_data():
    df = pd.read_csv("data/iris.csv")
    X_train, X_test, y_train, y_test = preprocess.preprocess_data(df)
    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0

def test_train_model():
    df = pd.read_csv("data/iris.csv")
    X_train, X_test, y_train, y_test = preprocess.preprocess_data(df)
    model = train.train_model(X_train, y_train)
    acc = train.evaluate_model(model, X_test, y_test)
    assert acc > 0.5


