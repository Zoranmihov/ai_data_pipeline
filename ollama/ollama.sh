#!/bin/bash

MODEL_NAME="deepseek-r1:7b"

echo "Starting Ollama daemon..."
ollama serve & sleep 5

echo "Checking for $MODEL_NAME in Ollama..."

# Check if the model is already downloaded
if ollama list | grep -q "$MODEL_NAME"; then
    echo "Model $MODEL_NAME is already available."
else
    echo "Model $MODEL_NAME not found. Downloading..."
    ollama pull $MODEL_NAME
    echo "Model $MODEL_NAME downloaded successfully."
fi

# Keep the container running (if needed)
echo "Ollama is running. Keeping container alive..."
tail -f /dev/null
