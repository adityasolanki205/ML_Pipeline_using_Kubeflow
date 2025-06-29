#!/usr/bin/env python
# coding: utf-8

# In[25]:


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


# In[26]:


@component(
    base_image="asia-south1-docker.pkg.dev/solar-dialect-264808/kubeflow-pipelines/demo_model"
)
def data_ingestion(input_data_path: str, input_data: Output[Dataset],):
    import pandas as pd
    from datetime import datetime, timedelta
    from google.cloud import bigquery
    import logging
    df = pd.read_csv(input_data_path)
    df.to_csv(input_data.path, index=False)


# In[27]:


@component(
    base_image="asia-south1-docker.pkg.dev/solar-dialect-264808/kubeflow-pipelines/demo_model"
)

def preprocessing(train_df: Input[Dataset], input_data_preprocessed: Output[Dataset]):
    import pandas as pd
    import numpy as np
    import logging
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import warnings
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    from sklearn.preprocessing import LabelEncoder

    def encode_columns(df, columns):
        encoders = {}
        for column in columns:
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])
            encoders[column] = dict(zip(le.transform(le.classes_), le.classes_))
        return df
    def preprocess(df):
        numeric_columns = df.describe().columns
        df_log_transformed = df.copy()
        df_log_transformed[numeric_columns] = df[numeric_columns].apply(lambda x: np.log(x + 1))
        scaler = MinMaxScaler()
        df_scaled_log_transformed = df_log_transformed.copy()
        df_scaled_log_transformed[numeric_columns] = scaler.fit_transform(df_scaled_log_transformed[numeric_columns])
        categorical_columns = [
        'Existing account', 'Credit history', 'Purpose', 'Saving',
        'Employment duration', 'Personal status', 'Debtors', 'Property',
        'Installment plans', 'Housing', 'Job', 'Telephone', 'Foreign worker'
        ]
        df_scaled_log_transformed = encode_columns(df_scaled_log_transformed, categorical_columns)
        return df_scaled_log_transformed

    df = pd.read_csv(train_df.path)
    df = preprocess(df)
    df.to_csv(input_data_preprocessed.path, index=False)


# In[28]:


@component(
    base_image="asia-south1-docker.pkg.dev/solar-dialect-264808/kubeflow-pipelines/demo_model"
)
def train_test_data_split(
    dataset_in: Input[Dataset],
    target_column: str,
    dataset_train: Output[Dataset],
    dataset_test: Output[Dataset],
    test_size: float = 0.2,
):
    import pandas as pd
    import logging
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    import pandas as pd
    from sklearn.utils import shuffle
    def get_train_test_splits(df, target_column, test_size_sample ):
        df = shuffle(df)
        x = df.drop(target_column, axis=1)
        y = df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = test_size_sample)

        X_train = pd.DataFrame(X_train)
        y_train = pd.DataFrame(y_train)
        X_test = pd.DataFrame(X_test)
        y_test = pd.DataFrame(y_test)
        X_train.reset_index(drop=True, inplace=True)
        y_train.reset_index(drop=True, inplace=True)
        X_test = X_test.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)
        X_train = pd.concat([X_train, y_train], axis=1)
        X_test = pd.concat([X_test, y_test], axis=1)
        X_train.columns = x.columns.to_list() + [target_column]
        X_test.columns = x.columns.to_list() + [target_column]
        return X_train, X_test
    data = pd.read_csv(dataset_in.path)
    X_train, X_test = get_train_test_splits(
        data, target_column, test_size
    )
    X_train.to_csv(dataset_train.path, index=False)
    X_test.to_csv(dataset_test.path, index=False)


# In[29]:


@component(
     base_image="asia-south1-docker.pkg.dev/solar-dialect-264808/kubeflow-pipelines/demo_model"
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
    from sklearn.linear_model import LogisticRegression
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
    space = {
        'C': hp.loguniform('C', -3, 3),  # log-uniform between ~0.05 to ~20
        'penalty': hp.choice('penalty', ['l1', 'l2']),  # safer to exclude 'elasticnet' unless solver == 'saga'
        'solver': hp.choice('solver', ['liblinear', 'saga']),  # only solvers that support l1
        'class_weight': hp.choice('class_weight', [None, 'balanced']),
        'max_iter': hp.choice('max_iter', [100, 1000,2500, 5000]),
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


# In[30]:


@component(
    base_image="asia-south1-docker.pkg.dev/solar-dialect-264808/kubeflow-pipelines/demo_model"
)
def deploy_model(
    project: str,
    region: str,
    ml_model: Input[Model],
    model_name: str,
    serving_container_image_uri: str,
    model_uri: Output[Artifact],
):
    from google.cloud import aiplatform
    import logging
    import os

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
    os.makedirs(model_uri.path, exist_ok=True)
    with open(os.path.join(model_uri.path, "model_uri.txt"), "w") as f:
        f.write(model.resource_name)


# In[31]:


@component(
    base_image="asia-south1-docker.pkg.dev/solar-dialect-264808/kubeflow-pipelines/demo_model"
)
def create_endpoint(
    project: str,
    region: str,
    model_name: str,
    model_uri: Input[Artifact],
):
    from google.cloud import aiplatform
    import logging
    import os
    with open(os.path.join(model_uri.path, "model_uri.txt"), "r") as f:
        model_resource_name = f.read()
    model = aiplatform.Model(model_resource_name)
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


# In[32]:


@dsl.pipeline(name="Training Pipeline", pipeline_root="gs://demo_bucket_kfl/pipeline_root_demo")
def pipeline(
    input_data_path: str = "gs://demo_bucket_kfl/clean_customer_data.csv",
    project_id: str = "solar-dialect-264808",
    region: str = "asia-south1",
    model_name: str = "demo_model",
    target: str = "Classification",
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
        model_name=model_name,
        model_uri = deploy_model_op.outputs["model_uri"]
    )
    create_endpoint_op.set_caching_options(False)
    
if __name__ == "__main__":
    compiler.Compiler().compile(pipeline_func=pipeline, package_path="training_pipeline.json")


# In[ ]:




