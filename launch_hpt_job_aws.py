import boto3
import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.tuner import (
    IntegerParameter,
    HyperparameterTuner
)

# Config
job_name = "xgboost-hpt-custom-105"
bucket = "proyecto-1-ml"
output_path = f"s3://{bucket}/output"
region = "us-east-1"
role = "arn:aws:iam::613602870396:role/SageMakerExecutionRole"

max_jobs = 40
max_parallel_jobs = 4

# URI de tu imagen Docker personalizada en ECR
image_uri = "613602870396.dkr.ecr.us-east-1.amazonaws.com/train-xgboost-custom:latest"

# Hiperparámetros constantes
static_hyperparams = {
    "year": "2025",
    "month": "6"
}

# Rango de hiperparámetros a tunear
hyperparameter_ranges = {
    "n_estimators": IntegerParameter(2, 20),
    "max_depth": IntegerParameter(2, 10)
}

# Estimador con imagen custom
estimator = Estimator(
    image_uri=image_uri,
    role=role,
    instance_count=1,
    instance_type="ml.m5.large",
    output_path=output_path,
    hyperparameters=static_hyperparams,
    sagemaker_session=sagemaker.Session()
)

# Tuner
tuner = HyperparameterTuner(
    estimator=estimator,
    objective_metric_name="f1_score",
    objective_type="Maximize",
    hyperparameter_ranges=hyperparameter_ranges,
    metric_definitions=[
        {"Name": "f1_score", "Regex": "f1_score: ([0-9\\.]+)"}
    ],
    max_jobs=max_jobs,
    max_parallel_jobs=max_parallel_jobs,
    base_tuning_job_name=job_name
)

# Lanzar job
tuner.fit(job_name=job_name)

print("✅ HPT job lanzado con imagen personalizada.")
