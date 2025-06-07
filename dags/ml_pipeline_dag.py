from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from pipeline import preprocess, train, evaluate

default_args = {
    "owner": "airflow",
    "start_date": datetime(2024, 1, 1),
    }

with DAG("ml_pipeline",
         default_args=default_args,
         schedule_interval="@daily",
         catchup=False) as dag:
    
    def preproccess_tesk():
        df = preprocess.load_data()
        X_train, X_test, y_train, y_test = preprocess.preprocess_data()
        # Save to Xcom
        return {
            "X_train": X_train.to_json(),
            "X_test": X_test.to_json(),
            "y_train": y_train.to_json(),
            "y_test": y_test.to_json()
        }

    def train_task(ti):
        import pandas as pd
        data = ti.xcom_pull(task_id="preprocess")
        X_train = pd.read_json(data["X_train"])
        X_test = pd.read_json(data["X_test"])
        y_train = pd.read_json(data["y_train"],typ="series")
        y_test = pd.read_json(data["y_test"], typ="series")

        model = train.train_model(X_train, y_train)
        train.evaluate_model(model, X_test, y_test)
        train.save_model(model)

    preprocess_op = PythonOperator(
        task_id="preprocess",
        python_callable=preprocess_task

    )

    train_op = PythonOperator(
        task_id="train",
        python_callable=train_task
    )

    preprocess_op >> train_op




