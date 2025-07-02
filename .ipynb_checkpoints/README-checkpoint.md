# ML Operations using Kubeflow
This is one of the **ML Operations** Repository. Here we will try to learn basics of Machine learning model deploement using **Kubeflow**. We will learn step by step how to create a MLOps pipeline using [German Credit Risk](https://www.kaggle.com/uciml/german-credit). The complete process is divided into multiple parts:

1. **Create Vertex AI workbench and Storage Bucket**
2. **Create a Artifact registry**
3. **Create Docker Image**
4. **Create Pipeline**

   4.a ***Ingest Data***

   4.b ***Preprocess Data***

   4.c ***Split Data into Training and Testing Dataset***

   4.d ***HyperParametering Tuning***

   4.e ***Deploy Model to Model Registry***

   4.f ***Create Endpoint for Online Prediction***

   4.g ***Defining the pipeline***
   
6. **Running the Pipeline**
7. **Metadata Management**


## Motivation
For the last few years, I have been part of a great learning curve wherein I have upskilled myself to move into a Machine Learning and Cloud Computing. This project was practice project for all the learnings I have had. This is first of the many more to come. 
 

## Libraries/frameworks used

<b>Built with</b>
- [Anaconda](https://www.anaconda.com/)
- [Python](https://www.python.org/)
- [Vertex AI](https://cloud.google.com/vertex-ai?hl=en)
- [Google Cloud Storage](https://cloud.google.com/storage)
- [Artifact Registry](https://cloud.google.com/artifact-registry/docs)
- [Cloud Build](https://cloud.google.com/build/docs)
- [Vertex AI Workbench](https://cloud.google.com/vertex-ai-notebooks?hl=en)
- [Vertex AI Model Registry](https://cloud.google.com/vertex-ai/docs/model-registry/introduction)
- [Vertex AI Online Prediction](https://cloud.google.com/vertex-ai/docs/predictions/get-predictions)
- [Vertex AI Metadata](https://cloud.google.com/vertex-ai/docs/ml-metadata/introduction)

## Cloning Repository

```bash
    # clone this repo:
    git clone https://github.com/adityasolanki205/ML_Pipeline_using_Kubeflow.git
```

## Pipeline Construction

Below are the steps to setup the enviroment and run the codes:

1. **Setup**: First we will have to setup free google cloud account which can be done [here](https://cloud.google.com/free). Then we need to Download the data from [German Credit Risk](https://www.kaggle.com/uciml/german-credit).

2. **Creating a input data**: Now we will create a input data on the Local Machine. This provides basic step to wrangle , preprocess and save the data. You can also refer this [notebook](https://github.com/adityasolanki205/ML_Pipeline_using_Kubeflow/blob/main/German%20Credit.ipynb). This also provide a process to create model on local machine

3. **Creating a Vertex AI Workbench and Cloud Storage bucket**: Here will we will create workbench and S3 bucket to be used in the process.

    - Goto to Vertex AI workbench
    - Select Instances, Click on Create New and create the instance in asia-south1 with default settings
    - After the instance becomes active, click on Juptyter Labs. Open a terminal anr run the below command.
    ```bash
       git clone https://github.com/adityasolanki205/ML_Pipeline_using_Kubeflow.git
       cd ML_Pipeline_using_Kubeflow
    ```

    - Goto to Storage Bucket
    - Click on create new and create a bucket with default setting in asia-south1 with the name 'demo-bucket-kfl'
    - Copy the file using 'gsutil cp clean_customer_data.csv gs://demo_bucket_kfl/'

https://github.com/user-attachments/assets/89739148-baa3-4d95-9371-b2bab4ae4ead

4. **Creating a Artifact Registry**: We will now create a Repository for our Docker Image to be stored. Process is provded below.

    - Goto to Artifact registry.
    - Click on create Repository, use default setting to create a Docker Repository in asia-south1 and the name
      'kubeflow-pipelines'


https://github.com/user-attachments/assets/76903e65-c08c-46f9-b86b-34de96268290



5. **Creating the Docker Image**: After creating the repository we will create the docker Image for Kubeflow Components. This will also install all the required libraries:

   - To create this image we go back to workbench.
   - Now we run the docker_build. This file contains all the commands to create the image. It also contains requirements.txt file to install all the dependancies.
     
    requirements.txt
    ```text
        pandas
        numpy
        scikit-learn
        joblib
        Cython
        hyperopt
        kfp
        db-dtypes
        
        # Google Cloud libraries
        google-cloud-aiplatform
        google-cloud-storage
        google-cloud-pubsub
        google-cloud-bigquery
        google-cloud-bigquery-storage
        googleapis-common-protos
    ```

    docker_build.sh
    ```bash
        FROM gcr.io/deeplearning-platform-release/base-cpu
        
        WORKDIR /
        COPY training_pipeline.py /
        COPY requirements.txt /
        COPY ./src/ /src
        RUN pip install --upgrade pip && pip install -r requirements.txt
    ```
    - To create the image, we run the command below

    ```bash
       bash docker_build.sh
    ```


https://github.com/user-attachments/assets/0ccded59-f2c7-4e1d-863b-c790e7eab21a



6. **Create Pipeline**: Now the real pipeline creating starts. Here will we will try to create pipeline components one by one. File to be used is training_pipeline.ipynb

   - ***Ingest Data*** : First step in the pipeline is Data ingestion. Here we simply read the data german_data.csv from our bucket. This method expect 2 arguments, one is input_data_path coming from input arguments of the python job and second is the output_dataset path to copy file output path

    ```python
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
    ```
    
    - ***Preprocess Data***: Second step in the pipeline is preprocess the data. Here we simply clean the data provided in output of previous step. This method expect 2 arguments, one is training_df coming from previous output of the python job and second is the output_dataset path to copy preprocessed file. Here we first convert reduce skewness in the data using logarithmic transform. Then we perform MinMAxscaling. At the end we perform label encoding to categorical columns. Then we write output data to output path.  

    ```python  
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
    ```
    
    - ***Split Data into Training and Testing Dataset***: Third step in the pipeline is split data into training and test datasets. We split the datasets train and test and save them to output paths. This methods has 5 arguments, one for dataset input from previous steps, label, output train and test dataset paths and test size. 
      
    ```python
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
    ```
    - ***HyperParametering Tuning***: Fourth step in the pipeline is perform hyperparameter tuning. Here we try to find the optimal settings for its parameters to improve model performance. We try to find best parameters for Logistic regression. 

    ```python
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
    ```
    
    - ***Deploy Model to Model Registry***: Fifth step in the pipeline is deploy the model to Model Registry. This will help us use the model for online prediction. Here we either deploy latest version of the model or a new model depending if the model is already existing. Here we have 6 arguments in the method, starting with project, region coming from input arguments, ml_model created in previous step, model_name to be deployed in, serving container image to be used to deploy the code, and model_uri to be used to create the Endpoint in next step.

    ```python
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
    ```
    
    - ***Create Endpoint for Online Prediction***: In the sixth step we simply create an endpoint to be used for online prediction. This method has project, region, model_name and model_uri as input arguments.

    ```python
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
        
    ```
    
    - ***Defining the pipeline***: At last we define the pipeline components.
      
    ```python
        @dsl.pipeline(name="Training Pipeline", pipeline_root="gs://demo_bucket_kfl/pipeline_root_demo")
        def pipeline(
            input_data_path: str = "gs://demo_bucket_kfl/german_data.csv",
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
    ```
    
7. **Running the Pipeline**: Now we simply have to run the pipeline. 

```python
    from google.cloud.aiplatform import PipelineJob
    
    pipeline_job = PipelineJob(
        display_name="training_pipeline_job",
        template_path="training_pipeline.json",
        pipeline_root="gs://demo_bucket_kfl/pipeline_root_demo",
        project="solar-dialect-264808",
        location="asia-south1",
        )
    pipeline_job.run()
```

https://github.com/user-attachments/assets/fb8a7cd2-2bd5-4127-8540-391058a45f8e


8. **Metadata Management**: At last we simply verify the metadata of the pipeline. The output artifacts are provided in output artifacts folder.


https://github.com/user-attachments/assets/944b2b5d-cf57-4817-bfa7-87f4496b55d6


10. **Delete Infrastructure (Optional)**: Please delete below mentioned services
    
    - Workbench
    - Storage Bucket
    - Delete Pipelines
    - Artifact Repository created in Artifact Registry
    - Undeploy the model from endpoint
    - Delete the endpoint
    - Delete model from model registry
    - Delete data from Metadata management


https://github.com/user-attachments/assets/5b765fcd-16e9-42e8-a9cf-2baf134e9f07


## Implementation
To test the code we need to do the following:

    1. Copy the repository in Cloud SDK using below command:
       git clone https://github.com/adityasolanki205/ML_Pipeline_using_Kubeflow.git
    
    2. Create a Storage Bucket by the name 'demo_bucket_kfl' in asia-south1
    
    3. Copy the data file in the cloud Bucket using the below command
        cd ML_Pipeline_using_Kubeflow
        gsutil cp german_data.csv gs://demo_bucket_kfl/
    
    4. Run the file training_pipeline.ipynb/ training_pipeline.py. This will craete a json file
    
    5. Run the run_pipeline.ipynb file
     
    6. Verify of all the artifacts are created.

## Credits
1. Akash Nimare's [README.md](https://gist.github.com/akashnimare/7b065c12d9750578de8e705fb4771d2f#file-readme-md)
2. [Building a Kubeflow Training Pipeline on Google Cloud: A Step-by-Step Guide](https://medium.com/@rajmudigonda893/building-a-kubeflow-training-pipeline-on-google-cloud-a-step-by-step-guide-761a6b0eb197)
3. [Deploy ML Training Pipeline Using Kubeflow](https://medium.com/@kavinduhapuarachchi/deploy-ml-training-pipeline-using-kubeflow-19d52d22f44f)
4. [A Beginner’s Guide to Kubeflow on Google Cloud Platform](https://medium.com/@vishwanath.prudhivi/a-beginners-guide-to-kubeflow-on-google-cloud-platform-5d02dbd2ec5e)
5. [MLOps 101 with Kubeflow and Vertex AI](https://medium.com/google-cloud/mlops-101-with-kubeflow-and-vertex-ai-61f6f5489fa8)
