import os
from datetime import datetime, timedelta
from pathlib import Path
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
import sys

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2026, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def run_stage_1():
    sys.path.insert(0, str(PROJECT_ROOT))
    from code.datasets.data_pipeline import run_data_engineering
    run_data_engineering(
        raw_path=str(PROJECT_ROOT / "data" / "raw" / "winequality-red.csv"),
        processed_dir=str(PROJECT_ROOT / "data" / "processed")
    )


def run_stage_2():
    sys.path.insert(0, str(PROJECT_ROOT))
    from code.models.train import run_model_engineering
    run_model_engineering(
        train_path=str(PROJECT_ROOT / "data" / "processed" / "train.csv"),
        test_path=str(PROJECT_ROOT / "data" / "processed" / "test.csv"),
        model_output_dir=str(PROJECT_ROOT / "models")
    )


with DAG(
    dag_id="wine_quality_mlops_pipeline",
    default_args=default_args,
    description="End-to-End Automated MLOps Pipeline for Wine Quality (Stages 1, 2, 3)",
    schedule_interval="*/5 * * * *",  # Run every 5 minutes
    catchup=False,
    max_active_runs=1,
    tags=["mlops", "wine_quality", "deployment"]
) as dag:

    stage_1_data_engineering = PythonOperator(
        task_id="stage_1_data_engineering",
        python_callable=run_stage_1,
    )

    stage_2_model_engineering = PythonOperator(
        task_id="stage_2_model_engineering",
        python_callable=run_stage_2,
    )

    stage_3_deployment = BashOperator(
        task_id="stage_3_deployment",
        bash_command=(
            f"cd {PROJECT_ROOT} && "
            "docker compose -f code/deployment/docker-compose.yml up -d --build"
        ),
    )
    stage_1_data_engineering >> stage_2_model_engineering >> stage_3_deployment
