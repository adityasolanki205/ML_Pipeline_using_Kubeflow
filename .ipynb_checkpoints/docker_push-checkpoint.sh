#!/bin/bash 
PROJECT_ID="solar-dialect-264808"
REGION="asia-south1"
REPOSITORY="kubeflow_pipelines"
IMAGE_TAG='demo_model:latest'


# Configure Docker
gcloud auth configure-docker $REGION-docker.pkg.dev
 
 # Push
docker push $REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/$IMAGE_TAG