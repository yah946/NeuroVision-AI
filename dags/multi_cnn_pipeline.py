#from datetime import datetime
#from airflow.decorators import task,dag
#from src.neuro_vision_ai import pipeline

#@dag(dag_id='multi_cnn_pipeline',start_date=datetime(2026,10,1),schedule=False,catchup=False)
#def taskflow_multi_cnn_pipeline():
#    @task
#    def pipeline_invoke():
#        pipeline()
#    pipeline_invoke()

#taskflow_multi_cnn_pipeline()

import os
import subprocess
from datetime import datetime

from airflow.decorators import dag, task


PROJECT_DIR = "/workspace/NeuroVision-AI"
ML_PYTHON = f"{PROJECT_DIR}/.venv/bin/python"


@dag(
    dag_id="multi_cnn_pipeline",
    start_date=datetime(2026, 10, 1),
    schedule=None,
    catchup=False,
)
def taskflow_multi_cnn_pipeline():

    @task
    def run_ml_pipeline():
        env = os.environ.copy()

        env["PYTHONPATH"] = (
            f"{PROJECT_DIR}/src"
            + os.pathsep
            + env.get("PYTHONPATH", "")
        )

        subprocess.run(
            [
                ML_PYTHON,
                "-c",
                (
                    "from neuro_vision_ai.pipeline "
                    "import pipeline; pipeline()"
                ),
            ],
            cwd=PROJECT_DIR,
            env=env,
            check=True,
        )

    run_ml_pipeline()


taskflow_multi_cnn_pipeline()
