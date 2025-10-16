#!/bin/bash

docker pull ghcr.io/open-webui/open-webui:main
docker run -d \
    --name open-webui \
    -p 8000:8000 \
    -p 3000:8080 \
    -v open-webui:/app/backend/data \
    -e OPENAI_API_BASE_URL=http://0.0.0.0:8000/v1 \
    -e OPENAI_API_KEY=empty \
    --restart always \
    ghcr.io/open-webui/open-webui:main