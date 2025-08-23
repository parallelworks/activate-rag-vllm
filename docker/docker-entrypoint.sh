#!/bin/bash
set -Eeuo pipefail

echo "Starting Chroma database service..."
chroma run --path /root/.cache/chroma --port $CHROMA_PORT >/logs/chroma.log 2>&1 & p1=$!

MAX_RETRIES=15   # number of attempts
echo "Waiting for Chroma on port $CHROMA_PORT for $MAX_RETRIES seconds..."

SLEEP_SECS=1     # wait time between retries
URL="http://127.0.0.1:$CHROMA_PORT/"
for i in $(seq 1 $MAX_RETRIES); do
  if curl -s "$URL" > /dev/null 2>&1; then
    echo "Chroma is up (attempt $i)"
    break   # leave the loop and continue the script
  fi
  echo "Attempt $i/$MAX_RETRIES: Chroma not ready, retrying in $SLEEP_SECS sec..."
  if [ "$i" -eq "$MAX_RETRIES" ]; then
    echo "Reached max retries ($MAX_RETRIES), exiting."
    exit 1
  fi
  sleep $SLEEP_SECS
done
sleep 1

echo "Running the rest of the services."
python3 /app/rag_server.py --port $RAG_PORT --collection activate_rag --embedding_model "${EMBEDDING_MODEL}" >/logs/rag_server.log 2>&1 & p2=$!
python3 /app/indexer.py --config /app/indexer_config.yaml --poll --rescan-seconds 10 >/logs/indexer.log 2>&1 & p3=$!

MAX_RETRIES=120   # number of attempts
echo "Waiting for VLLM on port $VLLM_SERVER_PORT for $MAX_RETRIES seconds..."

# wait for the vllm service to become available
SLEEP_SECS=1     # wait time between retries
URL="$VLLM_URL/models"
for i in $(seq 1 $MAX_RETRIES); do
  if curl -s "$URL" > /dev/null 2>&1; then
    echo "VLLM is up (attempt $i)"
    break   # leave the loop and continue the script
  fi
  echo "Attempt $i/$MAX_RETRIES: VLLM not ready, retrying in $SLEEP_SECS sec..."
  if [ "$i" -eq "$MAX_RETRIES" ]; then
    echo "Reached max retries ($MAX_RETRIES), exiting."
    exit 1
  fi
  sleep $SLEEP_SECS
done
python3 /app/rag_proxy.py >/logs/rag_proxy.log 2>&1 & p4=$!

echo "All services started."

trap 'kill -TERM $p1 $p2 $p3 $p4 2>/dev/null || true' TERM INT
wait -n $p1 $p2 $p3 $p4 # exits on any failure