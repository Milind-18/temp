from pipeline.preprocess import preprocess_data


def test_preprocess_data():
    X_train, X_test, y_train, y_test = preprocess_data()
    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
