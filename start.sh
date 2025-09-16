#!/bin/sh

WEIGHTS_URL="https://github.com/username/repo/releases/latest/weights.pt"
DEST_PATH="/app/weights.pt"

if [ ! -f "$DEST_PATH" ]; then
    echo "Downloading model weights..."
    curl -L -o "$DEST_PATH" "$WEIGHTS_URL"
fi

uvicorn tbank_logo_detector.service:app --host 0.0.0.0 --port 8000
