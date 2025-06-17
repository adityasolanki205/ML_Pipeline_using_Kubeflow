gcloud artifacts repositories create kubeflow_pipelines \
  --repository-format=docker \
  --location=asia-south1 \
  --description="Repository for Kubeflow pipeline container images"
