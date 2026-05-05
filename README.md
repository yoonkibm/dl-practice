# dl-practice

PyTorch-based vision framework practice project.

Goal:
- provide a timm-like model API, e.g. `model = vit("small", ...)`
- keep setup reproducible across different machines using `uv`
- build reusable block-level modules (residual, bottleneck, attention, etc.)

## Quick Start (after uv is installed)

```bash
uv sync --all-groups
uv run python --version
uv run pytest
```

## Project Structure

```text
src/
  dl_practice/
    modules/
      blocks/
        conv.py
        depth_wise_separable_conv.py
        residual.py
```

## Current Status

- `uv`-based dependency management is enabled (`pyproject.toml`, `uv.lock`).
- Base package uses `src` layout (`src/dl_practice`).
- Implemented modules:
  - `ConvBlock` in `src/dl_practice/modules/blocks/conv.py`
  - `DepthwiseSeparableConvBlock` in `src/dl_practice/modules/blocks/depth_wise_separable_conv.py`
  - `ResidualBlock` in `src/dl_practice/modules/blocks/residual.py`
    - supports `downsample` options: `None`/`"none"`, `"stride_conv"`, `"pool_proj"`

## Smoke Test Example

```bash
uv run python -c "import torch; from dl_practice.modules.blocks.conv import ConvBlock; x=torch.randn(2,3,32,32); print(ConvBlock(3,16)(x).shape)"
uv run python -c "import torch; from dl_practice.modules.blocks.depth_wise_separable_conv import DepthwiseSeparableConvBlock; x=torch.randn(2,8,32,32); print(DepthwiseSeparableConvBlock(8,16)(x).shape)"
uv run python -c "import torch; from dl_practice.modules.blocks.residual import ResidualBlock; x=torch.randn(2,64,56,56); print(ResidualBlock(64,64,None)(x).shape)"
```

## Development Note

- Main coding is done by the project owner.
- Codex is used for review, minimal fixes, and report generation under `task/reports`.
