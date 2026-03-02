# Environment

## System Requirements

- OS: Ubuntu 22.04
- Python: 3.11
- CUDA: 12.1
- cuDNN: 9.1.0
- GPU Driver: 570.195.03
- Tested on: NVIDIA L40S (46GB VRAM)

## Python Dependencies

Install dependencies using the venv:
```bash
python -m venv .venv_ff
.venv_ff/bin/pip install -r requirements.txt
```

For exact reproducibility use the lock file:
```bash
.venv_ff/bin/pip install -r requirements.lock
```

## Model Weights

Model weights are not included in this repository.
Download from OneDrive [link] and place in `.assets/models/`.

## Run
```bash
.venv_ff/bin/python facefusion.py run \
    --source <source_image> \
    --target <target_video> \
    --output-path <output_path> \
    --face-mask-types area \
    --execution-providers cuda
```
