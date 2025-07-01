#!/bin/bash 
PROJECT_ID="solar-dialect-264808"
REGION="asia-south1"
REPOSITORY="kubeflow-pipelines"
IMAGE='demo'
IMAGE_TAG='demo_model:latest'

#docker build -t $IMAGE .
gcloud builds --project $PROJECT_ID submit --tag $REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/$IMAGE_TAG .