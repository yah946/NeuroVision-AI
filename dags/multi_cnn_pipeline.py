from datetime import datetime
from airflow.decorators import task,dag
from src.neuro_vision_ai import pipeline
@dag(dag_id='multi_cnn_pipeline',start_date=datetime(2026,10,1),schedule=False,catchup=False)
def taskflow_multi_cnn_pipeline():
    @task
    def pipeline_invoke():
        pipeline()
    pipeline_invoke()

taskflow_multi_cnn_pipeline()
