# ML Operations using Kubleflow
This is one of the **ML Operations** Repository. Here we will try to learn basics of Machine learning model deploement using **Kubeflow**. We will learn step by step how to create a MlOps pipeline using [German Credit Risk](https://www.kaggle.com/uciml/german-credit). The complete process is divided into 6 parts:

1. **Create a Machine Learning Model**
2. **Create a Artifact registry**
3. **Create Docker Image**
4. **Create Pipeline**

   4.a ***Ingest Data***

   4.b ***Preprocess Data***

   4.c ***Split Data into Training and Testing Dataset***

   4.d ***HyperParametering Tuning***

   4.e ***Deploy Model to Model Registry***

   4.f ***Create Endpoint for Online Prediction***
   
6. **Running the Pipeline**
7. **Verifing the Artifacts**


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

## Cloning Repository

```bash
    # clone this repo:
    git clone https://github.com/adityasolanki205/ML_Pipeline_using_Kubeflow.git
```

## Pipeline Construction

Below are the steps to setup the enviroment and run the codes:

1. **Setup**: First we will have to setup free google cloud account which can be done [here](https://cloud.google.com/free). Then we need to Download the data from [German Credit Risk](https://www.kaggle.com/uciml/german-credit).

2. **Cloning the Repository to Cloud SDK**: We will have to copy the repository on Cloud SDK using below command:

```bash
    # clone this repo:
    git clone https://github.com/adityasolanki205/ML_Pipeline_using_Kubeflow.git
```

3. **Create a Machine Learning Model**: Now we will create a model on the Local Machine. Process to create that model is present [here](https://github.com/adityasolanki205/German-Credit.git). This provides basic step to wrangle , preprocess and save the data. You can also refer this [notebook](https://github.com/adityasolanki205/ML_Pipeline_using_Kubeflow/blob/main/German%20Credit.ipynb). 


4. **Create a Artifact Registry**: We will now create a Repository for our Docker Image to be stored. Process is provded below.

5. **Createing the Docker Image**: After creating the repository we will create the docker Image for Kubeflow Components. This will Also install all the required libraries:

docker_build.sh
```bash
    FROM gcr.io/deeplearning-platform-release/base-cpu

    WORKDIR /
    COPY training_pipeline.py /
    COPY requirements.txt /
    COPY ./src/ /src
    RUN pip install --upgrade pip && pip install -r requirements.txt
```

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

6. **Performing Type Convertion**: After Filtering we will convert the datatype of numeric columns from String to Int or Float datatype. Here we will use **Map()** to apply the Convert_Datatype(). 

```python
    ... 
    def Convert_Datatype(data):
        #This will convert the datatype of columns from String to integers or Float values
        data['Duration_month'] = float(data['Duration_month']) if 'Duration_month' in data else None
        data['Credit_amount'] = float(data['Credit_amount']) if 'Credit_amount' in data else None
        data['Installment_rate'] = float(data['Installment_rate']) if 'Installment_rate' in data else None
        data['Residential_Duration'] = float(data['Residential_Duration']) if 'Residential_Duration' in data else None
        data['Age'] = float(data['Age']) if 'Age' in data else None
        data['Number_of_credits'] = float(data['Number_of_credits']) if 'Number_of_credits' in data else None
        data['Liable_People'] = float(data['Liable_People']) if 'Liable_People' in data else None
        data['Existing_account'] =  int(data['Existing_account']) if 'Existing_account' in data else None
        data['Credit_history'] =  int(data['Credit_history']) if 'Credit_history' in data else None
        data['Purpose'] =  int(data['Purpose']) if 'Purpose' in data else None
        data['Saving'] =  int(data['Saving']) if 'Saving' in data else None
        data['Employment_duration'] =  int(data['Employment_duration']) if 'Employment_duration' in data else None
        data['Personal_status'] =  int(data['Personal_status']) if 'Personal_status' in data else None
        data['Debtors'] =  int(data['Debtors']) if 'Debtors' in data else None
        data['Property'] =  int(data['Property']) if 'Property' in data else None
        data['Installment_plans'] =  int(data['Installment_plans']) if 'Installment_plans' in data else None
        data['Housing'] =  int(data['Housing']) if 'Housing' in data else None
        data['Job'] =  int(data['Job']) if 'Job' in data else None
        data['Telephone'] =  int(data['Telephone']) if 'Telephone' in data else None
        data['Foreign_worker'] =  int(data['Foreign_worker']) if 'Foreign_worker' in data else None
        return data
    ...
    def run(argv=None, save_main_session=True):
        ...
        with beam.Pipeline(options=PipelineOptions()) as p:
             encoded_data = ( p 
                            | 'Read data' >> beam.io.ReadFromPubSub(topic=TOPIC).with_output_types(bytes) 
                            )
                    data =  ( encoded_data
                            | 'Decode' >> beam.Map(lambda x: x.decode('utf-8') 
                            ) 
              parsed_data = ( data 
                            | 'Parsing Data' >> beam.ParDo(Split())
                            )
           Converted_data = ( parsed_data
                            | 'Convert Datatypes' >> beam.Map(Convert_Datatype)
                            | 'Writing output' >> beam.io.WriteToText(known_args.output))

    if __name__ == '__main__':
        run()
```

7. **Predicting Customer segments**: Now we will implement the machine learning model. If you wish to learn how this machine learning model was created, please visit this [repository](https://github.com/adityasolanki205/German-Credit). We will save this model using JobLib library. To load the sklearn model we will have to follow the steps mentioned below:
    - Download the Model from Google Storage bucket using download_blob method
    
    - Load the model using setup() method in Predict_data() class
    
    - Predict Customer segments from the input data using Predict() method of sklearn
    
    - Add Prediction column in the output

```python
    ... 
    def call_vertex_ai(data):
    aiplatform.init(project='827249641444', location='asia-south1')
    feature_order = ['Existing_account', 'Duration_month', 'Credit_history', 'Purpose',
                 'Credit_amount', 'Saving', 'Employment_duration', 'Installment_rate',
                 'Personal_status', 'Debtors', 'Residential_Duration', 'Property', 'Age',
                 'Installment_plans', 'Housing', 'Number_of_credits', 'Job', 
                 'Liable_People', 'Telephone', 'Foreign_worker']
    endpoint = aiplatform.Endpoint(endpoint_name=f"projects/827249641444/locations/asia-south1/endpoints/6457541741091225600")
    features = [data[feature] for feature in feature_order]
    response = endpoint.predict(
        instances=[features]
    )
    
    prediction = response.predictions[0]
    data['Prediction'] = int(prediction)
    return data
    ...
    def run(argv=None, save_main_session=True):
        ...
        with beam.Pipeline(options=PipelineOptions()) as p:
            encoded_data   = ( p 
                             | 'Read data' >> beam.io.ReadFromPubSub(topic=TOPIC).with_output_types(bytes))
            data           = ( encoded_data
                             | 'Decode' >> beam.Map(lambda x: x.decode('utf-8')) 
            Parsed_data    = ( data 
                             | 'Parsing Data' >> beam.ParDo(Split()))
            Converted_data = ( Parsed_data
                             | 'Convert Datatypes' >> beam.Map(Convert_Datatype))
            Prediction     = ( Converted_data 
                             | 'Predition' >> 'Get Inference' >> beam.Map(call_vertex_ai))
            Output         = ( Prediction
                             | 'Saving the output' >> beam.io.WriteToText(known_args.output))
    if __name__ == '__main__':
        run()
```

10. **Inserting Data in Bigquery**: Final step in the Pipeline it to insert the data in Bigquery. To do this we will use **beam.io.WriteToBigQuery()** which requires Project id and a Schema of the target table to save the data. 

```python
    import apache_beam as beam
    from apache_beam.options.pipeline_options import PipelineOptions
    import argparse
    
    SCHEMA = 
    '
        Existing_account:INTEGER,
        Duration_month:FLOAT,
        Credit_history:INTEGER,
        Purpose:INTEGER,
        Credit_amount:FLOAT,
        Saving:INTEGER,
        Employment_duration:INTEGER,
        Installment_rate:FLOAT,
        Personal_status:INTEGER,
        Debtors:INTEGER,
        Residential_Duration:FLOAT,
        Property:INTEGER,
        Age:FLOAT,
        Installment_plans:INTEGER,
        Housing:INTEGER,
        Number_of_credits:FLOAT,
        Job:INTEGER,
        Liable_People:FLOAT,
        Telephone:INTEGER,
        Foreign_worker:INTEGER,
        Prediction:INTEGER
    '
    ...
    def run(argv=None, save_main_session=True):
        ...
        parser.add_argument(
          '--project',
          dest='project',
          help='Project used for this Pipeline')
        ...
        PROJECT_ID = known_args.project
        with beam.Pipeline(options=PipelineOptions()) as p:
            encoded_data   = ( p 
                             | 'Read data' >> beam.io.ReadFromPubSub(topic=TOPIC).with_output_types(bytes) 
                             )
            data           = ( encoded_data
                             | 'Decode' >> beam.Map(lambda x: x.decode('utf-8') 
                             ) 
            Parsed_data    = ( data 
                             | 'Parsing Data' >> beam.ParDo(Split()))
            Converted_data = ( Parsed_data
                             | 'Convert Datatypes' >> beam.Map(Convert_Datatype))

             Prediction     = ( Converted_data 
                             | 'Predition' >> 'Get Inference' >> beam.Map(call_vertex_ai))
            output         = ( Prediction      
                             | 'Writing to bigquery' >> beam.io.WriteToBigQuery(
                               '{0}:GermanCredit.GermanCreditTable'.format(PROJECT_ID),
                               schema=SCHEMA,
                               write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND)
                             )

    if __name__ == '__main__':
        run()        
```

## Implementation
To test the code we need to do the following:

    1. Copy the repository in Cloud SDK using below command:
    git clone https://github.com/adityasolanki205/ML-Streaming-pipeline-using-Dataflow.git
    
    2. Create a Storage Bucket by the name 'streaming-pipeline-testing' in us-east1 
    
    3. Create 2 separate subfolders temp and stage in the bucket
    
    4. Copy the machine learning model file in the cloud Bucket using the below command
    cd ML-Streaming-pipeline-using-Dataflow
    gsutil cp Selected_Model.pkl gs://streaming-pipeline-testing/
    
    5. Create a Dataset in us-east1 by the name GermanCredit
    
    6. Create a table in GermanCredit dataset by the name GermanCreditTable
    
    7. Create Pub Sub Topic by the name german_credit_data
    
    8. Install Apache Beam on the SDK using below command
    sudo pip3 install apache_beam[gcp]
    sudo pip3 install joblib
    sudo pip3 install sklearn
    
    9. Run the command and see the magic happen:
     python3 ml-streaming-pipeline.py   
         --runner DataFlowRunner   
         --project solar-dialect-264808   
         --bucket_name test_german_data   
         --temp_location gs://test_german_data/Batch/Temp   
         --staging_location gs://test_german_data/Batch/Stage   
         --region asia-south1   
         --job_name ml-stream-analysis   
         --input_subscription projects/solar-dialect-264808/subscriptions/german_credit_data-sub   
         --input_topic projects/solar-dialect-264808/topics/german_credit_data   
         --save_main_session True   
         --setup_file ./setup.py   
         --minNumWorkers 1   
         --maxNumWorkers 4   
         --streaming
     
    10. Open one more tab in cloud SDK and run below command 
    cd ML-Streaming-pipeline-using-Dataflow
    python3 publish_to_pubsub.py

## Credits
1. Akash Nimare's [README.md](https://gist.github.com/akashnimare/7b065c12d9750578de8e705fb4771d2f#file-readme-md)
2. [Apache Beam](https://beam.apache.org/documentation/programming-guide/#triggers)
3. [Building Data Processing Pipeline With Apache Beam, Dataflow & BigQuery](https://towardsdatascience.com/apache-beam-pipeline-for-cleaning-batch-data-using-cloud-dataflow-and-bigquery-f9272cd89eba)
4. [Let’s Build a Streaming Data Pipeline](https://towardsdatascience.com/lets-build-a-streaming-data-pipeline-e873d671fc57)
5. [Apache Beam + Scikit learn(sklearn)](https://medium.com/@niklas.sven.hansson/apache-beam-scikit-learn-19f8ad10d4d)
