# dl-practice

PyTorch-based vision framework project.

## Overview

This project aims to provide a timm-like user experience on top of PyTorch:

- simple model/block creation APIs
- reusable block-level modules (residual, bottleneck, attention, etc.)
- reproducible development environment with `uv`

## Requirements

- Python 3.11
- `uv` installed: https://docs.astral.sh/uv/getting-started/installation/

## Quick Start

```bash
git clone <your-repo-url>
cd dl-practice
uv sync --all-groups
uv run pytest
```

## Project Structure

```text
src/
  dl_practice/
    modules/
      blocks/
        residual.py
tests/
```

## Usage Example

```python
import torch
from dl_practice.modules.blocks.residual import ResidualBlock

x = torch.randn(2, 64, 56, 56)
block = ResidualBlock(64, 128, downsample="stride_conv")
y = block(x)
print(y.shape)  # torch.Size([2, 128, 28, 28])
```

## Implemented Module

- `ResidualBlock` (`src/dl_practice/modules/blocks/residual.py`)
- downsample options:
  - `None` or `"none"`: no spatial downsample
  - `"stride_conv"`: stride-2 projection shortcut
  - `"pool_proj"`: max-pool + 1x1 projection shortcut

## Run a Quick Smoke Test

```bash
uv run python -c "import torch; from dl_practice.modules.blocks.residual import ResidualBlock; x=torch.randn(2,64,56,56); print(ResidualBlock(64,64,None)(x).shape)"
```
