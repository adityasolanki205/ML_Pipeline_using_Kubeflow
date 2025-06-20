#!/usr/bin/env python
# coding: utf-8

# In[1]:


import yaml
from kfp import dsl
from kfp.dsl import (
    component,
    Metrics,
    Dataset,
    Input,
    Model,
    Artifact,
    OutputPath,
    Output,
)
from kfp import compiler
import google.cloud.aiplatform as aiplatform
import os


# In[2]:


@component(
    base_image="asia-south1-docker.pkg.dev/solar-dialect-264808/kubeflow_pipelines/demo_model"
)
def data_ingestion(input_data_path: str, input_data: Output[Dataset],):
    import pandas as pd
    from datetime import datetime, timedelta
    from google.cloud import bigquery
    import logging
    df = pd.read_csv(input_data_path)
    df.to_csv(input_data.path, index=False)


# In[3]:


@component(
    base_image="asia-south1-docker.pkg.dev/solar-dialect-264808/kubeflow_pipelines/demo_model"
)
def preprocessing(train_df: Input[Dataset], input_data_preprocessed: Output[Dataset]):
    import pandas as pd
    import numpy as np
    from src.preprocess import get_preprocessing
    import logging

    df = pd.read_csv(train_df.path)
    df = get_preprocessing(df)
    df.to_csv(input_data_preprocessed.path, index=False)


# In[4]:


@component(
    base_image="asia-south1-docker.pkg.dev/solar-dialect-264808/kubeflow_pipelines/demo_model"
)
def train_test_data_split(
    dataset_in: Input[Dataset],
    target_column: str,
    dataset_train: Output[Dataset],
    dataset_test: Output[Dataset],
    test_size: float = 0.2,
):
    from src.train_test_splits import get_train_test_splits
    import pandas as pd
    import logging

    data = pd.read_csv(dataset_in.path)
    X_train, X_test = get_train_test_splits(
        data, target_column, test_size
    )
    logger.info("Dataset train Shape: ", X_train.shape)
    logger.info("Dataset test Shape: ", X_test.shape)
    logger.info(f"X_train columns: {list(X_train.columns)}")
    X_train.to_csv(dataset_train.path, index=False)
    X_test.to_csv(dataset_test.path, index=False)


# In[5]:


@component(
     base_image="asia-south1-docker.pkg.dev/solar-dialect-264808/kubeflow_pipelines/demo_model"
)
def hyperparameters_training(
    dataset_train: Input[Dataset],
    dataset_test: Input[Dataset],
    target: str,
    max_evals: int,
    metrics: Output[Metrics],
    param_artifact: Output[Artifact],
    ml_model: Output[Model],
):
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
    import joblib
    import os
    import json
    import logging
    

    X_train = pd.read_csv(dataset_train.path)
    X_test = pd.read_csv(dataset_test.path)

    y_train = X_train[target]
    y_test = X_test[target]
    X_train = X_train.drop(target, axis=1)
    X_test = X_test.drop(target, axis=1)

    # Define the search space for Random Forest hyperparameters
    space = {
        'C' : hp.choice('C', [0.01,0.1, 1,10,100]), 
        'penalty' : hp.choice('penalty', ['l1', 'l2', 'elasticnet', 'none']), 
        'class_weight': hp.choice('class_weight', ['balanced']),
        'solver' :  hp.choice('solver' , ['lbfgs','newton-cg','liblinear','sag','saga']),
        'max_iter' :  hp.choice('max_iter', [100, 1000,2500, 5000]),
    }
    def objective(params):
        rf = LogisticRegression(**params)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        metrics.log_metric("accuracy", accuracy)
        metrics.log_metric("precision", precision)
        metrics.log_metric("recall", recall)
        metrics.log_metric("f1", f1)

        return {'loss': -accuracy, 'status': STATUS_OK, 'model': rf}
    trials = Trials()
    
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=max_evals, trials=trials)

    best_params = trials.best_trial['result']['model'].get_params()
    best_model = trials.best_trial['result']['model']

    # Save the best model
    os.makedirs(ml_model.path, exist_ok=True)
    joblib.dump(best_model, os.path.join(ml_model.path, 'model.joblib'))

    # Save the best hyperparameters
    with open(param_artifact.path, "w") as f:
        json.dump(best_params, f)


# In[6]:


@component(
    base_image="asia-south1-docker.pkg.dev/solar-dialect-264808/kubeflow_pipelines/demo_model"
)
def deploy_model(
    project: str,
    region: str,
    ml_model: Input[Model],
    model_name: str,
    serving_container_image_uri: str,
):
    from google.cloud import aiplatform
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    existing_models = aiplatform.Model.list(
        filter=f"display_name={model_name}", project=project, location=region
    )
    if existing_models:
        latest_model = existing_models[0]
        logger.info(f"Creating a new version for existing model: {latest_model.name}")
        model = aiplatform.Model.upload(
            display_name=model_name,
            artifact_uri=ml_model.path,
            location='asia-south1',
            serving_container_image_uri=serving_container_image_uri,
            parent_model=latest_model.resource_name,
        )
    else:
        logger.info("No existing model found. Creating a new model.")
        model = aiplatform.Model.upload(
            display_name=model_name,
            artifact_uri=ml_model.path,
            location='asia-south1',
            serving_container_image_uri=serving_container_image_uri,
        )

        


# In[7]:


@component(
    base_image="asia-south1-docker.pkg.dev/solar-dialect-264808/kubeflow_pipelines/demo_model"
)
def create_endpoint(
    project: str,
    region: str,
    ml_model: Input[Model],
    model_name: str,
    serving_container_image_uri: str,
):
    from google.cloud import aiplatform
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    
    traffic_split = {"0": 100}
    machine_type = "n1-standard-4"
    min_replica_count = 1
    max_replica_count = 1
    
    endpoint = model.deploy(
            deployed_model_display_name=model_name,
            machine_type=machine_type,
            traffic_split = traffic_split,
            min_replica_count=min_replica_count,
            max_replica_count=max_replica_count
        )

        


# In[8]:


@dsl.pipeline(name="Training Pipeline", pipeline_root="gs://demo_bucket_kfl/pipeline_root_demo")
def pipeline(
    input_data_path: str = "gs://demo_bucket_kfl/clean_customer_data.csv",
    project_id: str = "solar-dialect-264808",
    region: str = "asia-south1",
    model_name: str = "demo_model",
    target: str = "type",
    max_evals: int = 30,
    use_hyperparameter_tuning: bool = True,
    serving_container_image_uri: str = "asia-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-5:latest"
):
    data_op = data_ingestion(
        input_data_path=input_data_path)
    data_op.set_caching_options(False)

    data_preprocess_op = preprocessing(train_df=data_op.outputs["input_data"])
    data_preprocess_op.set_caching_options(False)
    train_test_split_op = train_test_data_split(
        dataset_in=data_preprocess_op.outputs["input_data_preprocessed"],
        target_column="Classification",
        test_size=0.2,
    )
    train_test_split_op.set_caching_options(False)
    hyperparam_tuning_op = hyperparameters_training(
        dataset_train=train_test_split_op.outputs["dataset_train"],
        dataset_test=train_test_split_op.outputs["dataset_test"],
        target=target,
        max_evals=max_evals
    )
    hyperparam_tuning_op.set_caching_options(False)
    deploy_model_op = deploy_model(
        project=project_id, region=region,
        ml_model=hyperparam_tuning_op.outputs["ml_model"],
        model_name=model_name,
        serving_container_image_uri=serving_container_image_uri
    )
    deploy_model_op.set_caching_options(False)
    create_endpoint_op = create_endpoint(
        project=project_id, region=region,
        ml_model=hyperparam_tuning_op.outputs["ml_model"],
        model_name=model_name,
        serving_container_image_uri=serving_container_image_uri
    )
    create_endpoint_op.set_caching_options(False)
    

if __name__ == "__main__":
    compiler.Compiler().compile(pipeline_func=pipeline, package_path="training_pipeline.json")


# In[ ]:




