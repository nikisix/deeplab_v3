ENDPOINT_ID="3811910056476147712"
PROJECT_ID="six-analytics"
INPUT_DATA_FILE="./owl-toadstuhl.jpg.json"

curl \
-X POST \
-H "Authorization: Bearer $(gcloud auth print-access-token)" \
-H "Content-Type: application/json" \
https://us-central1-prediction-aiplatform.googleapis.com/v1alpha1/projects/${PROJECT_ID}/locations/us-central1/endpoints/${ENDPOINT_ID}:predict \
-d "@${INPUT_DATA_FILE}"
