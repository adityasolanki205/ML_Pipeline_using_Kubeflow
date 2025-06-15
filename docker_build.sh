#!/bin/bash 
PROJECT_ID="solar-dialect-264808"
REGION="asia-south1"
REPOSITORY="kubeflow_pipelines"
IMAGE='demo'
IMAGE_TAG='demo_model:latest'

docker build -t $IMAGE .
docker tag $IMAGE $REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/$IMAGE_TAG