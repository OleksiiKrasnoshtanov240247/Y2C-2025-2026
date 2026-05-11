#!/bin/bash
set -e

echo "================================================"
echo "Finishing FaceFusion Installation"
echo "================================================"

echo ""
echo "Enter your FaceFusion directory path:"
read FACEFUSION_DIR

cd $FACEFUSION_DIR

echo ""
echo "Installing remaining dependencies (skipping torch/torchvision)..."

# Install each package individually, skipping torch/torchvision
.venv_facefusion/bin/pip install \
    aiofiles==24.1.0 \
    annotated-doc==0.0.4 \
    annotated-types==0.7.0 \
    anyio==4.12.1 \
    brotli==1.2.0 \
    certifi==2026.1.4 \
    charset-normalizer==3.4.4 \
    click==8.3.1 \
    colorama==0.4.6 \
    coloredlogs==15.0.1 \
    fastapi==0.128.2 \
    ffmpy==1.0.0 \
    filelock==3.20.3 \
    flatbuffers==25.12.19 \
    fsspec==2026.2.0 \
    gradio==5.44.1 \
    gradio-client==1.12.1 \
    gradio-rangeslider==0.0.8 \
    groovy==0.1.2 \
    h11==0.16.0 \
    httpcore==1.0.9 \
    httpx==0.28.1 \
    huggingface-hub==0.36.2 \
    humanfriendly==10.0 \
    idna==3.11 \
    jinja2==3.1.6 \
    markdown-it-py==4.0.0 \
    markupsafe==3.0.3 \
    mdurl==0.1.2 \
    ml-dtypes==0.5.4 \
    mpmath==1.3.0 \
    networkx==3.6.1 \
    numpy==2.2.6 \
    onnx==1.19.1 \
    onnxruntime==1.23.2 \
    onnxruntime-gpu==1.24.1 \
    opencv-python==4.12.0.88 \
    orjson==3.11.7 \
    packaging==26.0 \
    pandas==2.3.3 \
    pillow==11.3.0 \
    protobuf==6.33.5 \
    psutil==7.1.3 \
    pydantic==2.11.10 \
    pydantic-core==2.33.2 \
    pydub==0.25.1 \
    pygments==2.19.2 \
    pyreadline3==3.5.4 \
    python-dateutil==2.9.0.post0 \
    python-multipart==0.0.22 \
    pytz==2025.2 \
    pyyaml==6.0.3 \
    requests==2.32.5 \
    rich==14.3.2 \
    ruff==0.15.0 \
    safehttpx==0.1.7 \
    scipy==1.16.3 \
    semantic-version==2.10.0 \
    setuptools==70.2.0 \
    shellingham==1.5.4 \
    six==1.17.0 \
    starlette==0.50.0 \
    sympy==1.13.1 \
    tomlkit==0.13.3 \
    tqdm==4.67.1 \
    typer==0.21.1 \
    typing-extensions==4.15.0 \
    typing-inspection==0.4.2 \
    tzdata==2025.3 \
    urllib3==2.6.3 \
    uvicorn==0.40.0 \
    websockets==15.0.1

echo ""
echo "✓ All dependencies installed!"

echo ""
echo "Verifying..."
.venv_facefusion/bin/python << PYEOF
import torch
import gradio
import onnxruntime

print("\n✅ FaceFusion Ready!")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
print(f"Gradio: {gradio.__version__}")
PYEOF

echo ""
echo "================================================"
echo "✅ Complete! Test with:"
echo "  cd $FACEFUSION_DIR"
echo "  .venv_facefusion/bin/python facefusion.py run --open-browser"
echo "================================================"