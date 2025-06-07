from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from pipeline import preprocess, train, evaluate

default_args = {"owner": "airflow", "start_date": datetime(2023, 1, 1)}

with DAG("ml_pipeline", schedule_intervals="@daily", default_args=default_args, catchup=False) as dag:

    def step_preprocess(ti):
        X_train, X_test, y_train, y_test = preprocess.preprocess_data()
        ti.xcom_push("X_train", X_train)
        ti.xcom_push("X_test",X_test)
        ti.xcom_push("y_train",y_train)
        ti.xcom_push("y_test",y_test)

    def step_train(ti):
        X_train = ti.xcom_pull(task_ids="preprocess", key="X_train")
        y_train = ti.xcom_pull(task_ids="preprocess", key="y_train")
        model = train.train_model(X_train, y_train)
        ti.xcom_push("model", model)

    def step_evaluate(ti):
        model = ti.xcom_pull(task_ids="train", key="model")
        X_test = ti.xcom_pull(task_ids="preprocess", key="X_test")
        y_test = ti.xcom_pull(task_ids="preprocess", key="y_test")
        acc = evaluate.evaluate_model(model, X_test, y_test)
        print(f"Accuracy: {acc}")

    t1 = PythonOperator(task_id="preprocess", python_callable=step_preprocess)
    t2 = PythonOperator(task_id="train", python_callable=step_train)
    t3 = PythonOperator(task_id="evaluate", python_callable=step_evaluate)

    t1 >> t2 >> t3


