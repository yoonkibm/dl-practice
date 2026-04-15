# dl-practice

PyTorch-based vision framework practice project.

Goal:
- provide a timm-like model API, e.g. `model = vit("small", ...)`
- keep setup reproducible across different machines using `uv`

## Quick Start (after uv is installed)

```bash
uv sync --all-groups
uv run python --version
uv run pytest
```
