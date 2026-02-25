#!/bin/bash
set -e

echo "================================================"
echo "FaceFusion COMPLETE Setup - ONE Script"
echo "================================================"

# Get directory
echo ""
echo "Enter FaceFusion directory (e.g., /home/y2b/facefusion):"
read FF_DIR
cd $FF_DIR

# Install system deps
echo ""
echo "Installing system dependencies..."
apt-get update && apt-get install -y python3.11 python3.11-venv ffmpeg

# Create venv
echo ""
echo "Creating venv..."
python3.11 -m venv .venv_ff
.venv_ff/bin/pip install --upgrade pip wheel

# Install PyTorch (do NOT reinstall later)
echo ""
echo "Installing PyTorch..."
.venv_ff/bin/pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# Install ALL other dependencies
echo ""
echo "Installing dependencies..."
.venv_ff/bin/pip install aiofiles annotated-doc annotated-types anyio brotli certifi charset-normalizer click colorama coloredlogs fastapi ffmpy flatbuffers gradio==5.44.1 gradio-client gradio-rangeslider groovy h11 httpcore httpx huggingface-hub humanfriendly idna markdown-it-py markupsafe mdurl ml-dtypes networkx numpy onnx onnxruntime onnxruntime-gpu opencv-python orjson pandas pillow protobuf psutil pydantic pydantic-core pydub pygments python-dateutil python-multipart pytz pyyaml requests rich ruff safehttpx scipy semantic-version shellingham starlette tomlkit tqdm typer typing-inspection tzdata urllib3 uvicorn websockets

echo ""
echo "✅ Done!"
echo ""
echo "Test:"
echo "  cd $FF_DIR"
echo "  .venv_ff/bin/python facefusion.py run --open-browser"