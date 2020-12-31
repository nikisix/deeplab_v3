MODEL_NAME="segmenter"
VERSION_NAME="v1"
REGION="us-central1"
INPUT_DATA_FILE="./tmp_input/owl-toadstuhl.jpg.json"

gcloud ai-platform predict \
    --model=$MODEL_NAME \
    --version=$VERSION_NAME \
    --json-request=$INPUT_DATA_FILE \
    --region=$REGION
