#!/bin/sh

OWNER="dangooddd"
REPO="tbank-sirius-cv"
ASSET_NAME="weights.pt"
DEST_PATH="weights/weights.pt"

if [ ! -f "$DEST_PATH" ]; then
    echo "Downloading model weights..."

    DOWNLOAD_URL=$(curl -s https://api.github.com/repos/"$OWNER"/"$REPO"/releases/latest \
      | jq -r ".assets[] | select(.name==\"$ASSET_NAME\") | .url")

    mkdir -p "$(dirname "$DEST_PATH")"

    if ! curl -L -H "Accept: application/octet-stream" "$DOWNLOAD_URL" -o "$DEST_PATH"; then
        echo "Error: Failed to download model weights" >&2
        exit 1
    fi
else
    echo "Weights already downloaded."
fi

uv run uvicorn tbank_logo_detector.service:app --host 0.0.0.0 --port 8000
