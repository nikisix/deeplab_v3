MODEL_DIR="gs://six-ai-platform-models/deeplabv3_sthalles"
VERSION=3
VERSION_NAME="v$VERSION"
MODEL_NAME="segmenter"
FRAMEWORK="tensorflow"
MACHINE_TYPE="mls1-c1-m2"
PYTHON_VERSION="3.7"  # actually 3.6
RUNTIME_VERSION="1.15"
REGION="us-central1"

# gcloud auth login
# gcloud ai-platform models create $MODEL_NAME
# gcloud config set ai_platform/region us-central1

gsutil mb $MODEL_DIR
gsutil rm -r gs://six-ai-platform-models/deeplabv3_sthalles
gsutil cp -r ./versions/$VERSION gs://six-ai-platform-models/deeplabv3_sthalles

gcloud ai-platform versions create $VERSION_NAME \
  --model=$MODEL_NAME \
  --origin=$MODEL_DIR \
  --runtime-version=$RUNTIME_VERSION \
  --framework=$FRAMEWORK \
  --python-version=$PYTHON_VERSION \
  --region=$REGION
# --machine-type=$MACHINE_TYPE

# view with
MODEL_NAME="segmenter"
gcloud ai-platform versions list --model=$MODEL_NAME
