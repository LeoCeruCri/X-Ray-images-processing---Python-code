# X-Ray-images-processing---Python-code

Python code for processing X-ray radiographs of a bubbling fluidized bed with
3D-printed distributor plates, recorded with the TU Delft set-up of three
stationary X-ray source–detector pairs.

**Status:** phase 1 (2D, per projection). The 3D tomographic reconstruction
comes in phase 2.

## Pipeline (phase 1)

| Step | Module | Status |
|---|---|---|
| Read TIFF frames (read-only) | `xrayfb.io` | done |
| Dark subtraction | `xrayfb.corrections` | done |
| Dead-pixel interpolation | `xrayfb.corrections` | generic mask version; detector line map pending |
| Cross-scatter correction | – | pending |
| Two-point holdup (empty / packed bed) | `xrayfb.holdup` | done (uncorrected) |
| Beam-hardening and dynamic-bias correction | – | pending |
| Bubble detection and properties | – | pending |

## Installation (Windows, VS Code)

Requires Python 3.14 (python.org installer).

```powershell
git clone https://github.com/LeoCeruCri/X-Ray-images-processing---Python-code.git
cd X-Ray-images-processing---Python-code
py -3.14 -m venv .venv
.venv\Scripts\activate
python -m pip install -e ".[dev]"
pre-commit install          # strips notebook outputs on commit
pytest                      # all tests must pass
```

VS Code picks up `.venv` and the tests automatically (`.vscode/settings.json`).

## Usage

1. Copy `configs/example_run.yaml` to `configs/local/<run>.yaml` (ignored by
   git) and set the paths.
2. Run `python scripts/quick_holdup.py configs/local/<run>.yaml`.

## Data rules

- Raw data are never modified: the code only reads them, and `io.write_image`
  refuses to write inside `raw_root`.
- No experimental data in git (`*.tif` is ignored).
- Every parameter goes in a YAML file under `configs/`, none in the code.

## Third-party code

- [RikVolger/3ct-tools](https://github.com/RikVolger/3ct-tools) (MIT): reference
  for the preprocessing steps; adapted code is credited where it is used.
- [RikVolger/fluidized-bed-ct](https://github.com/RikVolger/fluidized-bed-ct) and
  [CaTE](https://github.com/RikVolger/CaTE) (GPL-3.0): planned as external
  dependencies for phase 2; their code is not copied into this repository.
