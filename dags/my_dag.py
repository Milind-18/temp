from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from pathlib import Path

def my_task():
    print("Hello from Airflow!")

with DAG(
        dag_id="my_dag",
        start_date=datetime(2023,1,1),
        schedule_interval="@daily",
        catchup=False,
        tags=["custom"]
) as dag:
    
    run_my_task= PythonOperator(
        task_id="run_my_task",
        python_callable=my_task
    )
