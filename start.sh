#!/bin/sh

URL="https://github.com/dangooddd/tbank-sirius-cv/releases/latest/download/weights.pt"
DEST_PATH="weights/weights.pt"

if [ ! -f "$DEST_PATH" ]; then
    echo "Downloading model weights..."
    mkdir -p "$(dirname "$DEST_PATH")"

    if ! wget -T 10 -t 1 "$URL" -O "$DEST_PATH"; then
        echo "Error: Failed to download model weights." >&2
        exit 1
    fi
else
    echo "Weights already downloaded."
fi

uv run --no-dev uvicorn tbank_logo_detector.service:app --host 0.0.0.0 --port 8000
