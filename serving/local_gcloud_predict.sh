version_num=3
# rm -r ./versions/$version_num
gcloud ai-platform local predict \
    --model-dir ./versions/$version_num/ \
    --json-request ./tmp_input/owl-toadstuhl.jpg.json \
    --framework tensorflow \
    2>&1 | gsed -e 's/:/:\n/g' -e 's/\\n/\n/g'
