"""
### Airflow Lab 1: breast cancer classifier

Trains a RandomForest on the Wisconsin breast cancer dataset and registers the model
only if its test F1 clears a threshold.

`load_data → preprocess → train_model → evaluate_model → check_quality → register_model | reject_model → predict`

`predict` classifies the unlabeled tumors in `data/test.csv` with the registered model.

"""
# Import necessary libraries and modules
from datetime import datetime, timedelta

from airflow.providers.standard.operators.python import BranchPythonOperator, PythonOperator
from airflow.sdk import DAG, Param

from src.breast_cancer_classification import evaluate_model, load_data, predict, preprocess, register_model, reject_model, train_model

# Define default arguments for your DAG
default_args = {
    'owner': 'Rex Remigius Stephen Jothi',
    'start_date': datetime(2026, 10, 30),
    'retries': 0,  # Number of retries in case of task failure
    'retry_delay': timedelta(minutes=5),  # Delay before retries
}


def choose_branch(metrics: dict, min_f1: float) -> str:
    """Returns the task_id to run next. Airflow skips the other branch."""
    if metrics['f1'] >= min_f1:
        print(f"F1 {metrics['f1']} >= {min_f1}, registering the model")
        return 'register_model_task'
    print(f"F1 {metrics['f1']} < {min_f1}, rejecting the model")
    return 'reject_model_task'


with DAG(
    'Airflow-Lab',
    default_args=default_args,
    description='Train a breast cancer classifier and register it only if it clears an F1 threshold',
    schedule='@daily',  
    catchup=False,
    params={
        'n_estimators': Param(100, type='integer', minimum=1, description='Number of trees in the forest'),
        'max_depth': Param(5, type='integer', minimum=1, description='Maximum depth of each tree'),
        'min_f1': Param(0.90, type='number', minimum=0, maximum=1, description='Register the model only if its test F1 is at least this'),
    },
    render_template_as_native_obj=True,
    tags=['breast_cancer', 'ml'],
    doc_md=__doc__,
) as dag:

    run_kwargs = {'run_id': '{{ run_id }}'}

    # Task to validate the CSV and snapshot it for this run
    load_data_task = PythonOperator(
        task_id='load_data_task',
        python_callable=load_data,
        op_kwargs=run_kwargs,
    )

    # Task to clean the data and write the train/test split
    preprocess_task = PythonOperator(
        task_id='preprocess_task',
        python_callable=preprocess,
        op_kwargs=run_kwargs,
    )

    # Task to train the RandomForest with the hyperparameters chosen at trigger time
    train_model_task = PythonOperator(
        task_id='train_model_task',
        python_callable=train_model,
        op_kwargs={
            **run_kwargs,
            'n_estimators': '{{ params.n_estimators }}',
            'max_depth': '{{ params.max_depth }}',
        },
    )

    # Task to score the model; its return value (the metrics dict) is pushed to XCom
    evaluate_model_task = PythonOperator(
        task_id='evaluate_model_task',
        python_callable=evaluate_model,
        op_kwargs=run_kwargs,
    )

    # Branch task that pulls the metrics from XCom and picks which task runs next
    check_quality_task = BranchPythonOperator(
        task_id='check_quality_task',
        python_callable=choose_branch,
        op_kwargs={'metrics': evaluate_model_task.output, 'min_f1': '{{ params.min_f1 }}'},
    )

    # Task to copy the accepted model into the registry
    register_model_task = PythonOperator(
        task_id='register_model_task',
        python_callable=register_model,
        op_kwargs=run_kwargs,
    )

    # Task to log why the model was rejected
    reject_model_task = PythonOperator(
        task_id='reject_model_task',
        python_callable=reject_model,
        op_kwargs={**run_kwargs, 'min_f1': '{{ params.min_f1 }}'},
    )

    # Task to classify the new tumors in data/test.csv with the registered model.
    # One branch is always skipped, so the default trigger rule (all_success) would skip this too
    predict_task = PythonOperator(
        task_id='predict_task',
        python_callable=predict,
        op_kwargs=run_kwargs,
        trigger_rule='none_failed_min_one_success',
    )

    # Set task dependencies
    load_data_task >> preprocess_task >> train_model_task >> evaluate_model_task >> check_quality_task
    check_quality_task >> [register_model_task, reject_model_task] >> predict_task

if __name__ == "__main__":
    dag.test()
