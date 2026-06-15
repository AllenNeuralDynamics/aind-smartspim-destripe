# aind-smartspim-destripe

Removes horizontal stripe artifacts from lightsheet microscopy images acquired with the SmartSPIM microscope. The pipeline uses a **log-space Fast Fourier Transform (FFT)** filter combined with flatfield correction and is designed to operate on large-scale Zarr datasets.

Developed and maintained by the [Allen Institute](https://www.alleninstitute.org/what-we-do/brain-science/research/allen-institute-for-neural-dynamics/).

---

## Background

SmartSPIM lightsheet imaging of cleared brain tissue generates horizontal stripe artifacts caused by scattering and absorption inhomogeneities along the light sheet. Standard dual-band filtering suppresses these stripes but introduces ringing artifacts around high-intensity structures (e.g., fluorescently labeled cell bodies), because the bright signal bleeds into the filtered frequency bands.

The log-space FFT approach addresses this by:
1. Compressing the dynamic range with a log transform, which elevates background noise and reduces the dominance of bright cells.
2. Isolating stripe-related frequencies via wavelet decomposition and directional FFT filtering.
3. Applying the correction selectively — using gentler parameters for slices that contain cells and more aggressive parameters for background-only slices.

---

## Algorithm

Each image plane is processed as follows:

1. **Log transform** — expands the low-intensity range so background stripes become more prominent relative to signal.
2. **2D wavelet decomposition** (Daubechies-3) — separates the image into subbands; horizontal detail coefficients capture stripe energy.
3. **FFT along the stripe direction** — transforms horizontal coefficients into the frequency domain.
4. **Otsu threshold + Gaussian notch filter** — masks stripe frequencies while preserving image structure frequencies.
5. **Inverse FFT + inverse wavelet** — reconstructs the destriped plane.
6. **Reverse log transform** — restores original intensity scale.
7. **Flatfield / darkfield correction** — removes vignetting and uneven illumination using either:
   - **Retrospective**: flatfield estimated from the data itself via [BaSiC](https://github.com/peng-lab/BaSiCPy).
   - **Prospective**: pre-computed flatfields from the microscope, matched to each tile by position and emission wavelength.

**Dual-mode filtering** — before filtering, the algorithm checks whether a slice contains cells (foreground mean > background mean and mean intensity > x_intensity). Two parameter sets are used:

| Mode | Wavelet | Sigma | Max threshold |
|------|---------|-------|---------------|
| No cells (background) | db3 | 128 | 12 |
| With cells | db3 | 64 | 3 |

---

## Pipeline Architecture

```
run_capsule.py
└── zarr_destriper.destripe_channel()        # per emission channel
    └── zarr_destriper.destripe_zarr()       # chunked processing
        ├── Data loader (super-chunks: 384×1600×2000 voxels)
        ├── Multiprocessing workers
        │   └── filtering.filter_stripes()   # core algorithm per chunk
        └── blocked_zarr_writer              # Zarr output (uint16, Blosc zstd)
```

- **Entry point**: `code/run_capsule.py`
- **Channels**: discovered from `Ex_*_Em_*` folders or S3 paths
- **Output format**: OME-Zarr with a 3-level multiscale pyramid (factor-2 downsampling per spatial axis)
- **Metadata**: `processing.json` (AIND processing schema)

---

## Installation

```bash
# User install
pip install -e .

# Development install (linters + tests)
pip install -e .[dev]
```

---

## Requirements

| Package | Version |
|---------|---------|
| Python | 3.10 |
| numpy | 1.26.4 |
| scipy | 1.13.1 |
| scikit-image | 0.24.0 |
| BaSiCPy | 1.1.0 |
| jax / jaxlib | 0.4.23 |
| PyWavelets | 1.6.0 |
| natsort | 8.4.0 |
| aind-data-schema | 2.8.0 |
| aind-data-schema-models | >=5.7.1,<6 |
| boto3 / s3fs | latest |

---

## Usage

### Full pipeline (Code Ocean capsule)

The pipeline expects the following layout under `../data/`:

```
data/
├── acquisition.json
├── data_description.json
├── preprocess_<channel>.json
├── laser_tiles.json
├── estimated_flat_laser_<channel>*.tif   # per-channel flatfields
└── derivatives/                          # optional prospective flats
    ├── FlatReal*.tif
    └── DarkMaster_cropped.tif
```

Run:

```bash
cd code
python -u run_capsule.py
```

### Programmatic — full Zarr channel

```python
from aind_smartspim_destripe.zarr_destriper import destripe_channel

destripe_channel(
    channel_path="/path/to/channel",
    output_path="/path/to/output",
    no_cells_config={"wavelet": "db3", "level": None, "sigma": 128, "max_threshold": 12},
    cells_config={"wavelet": "db3", "level": None, "sigma": 64, "max_threshold": 3},
    retrospective=True,
)
```

### Programmatic — batch image files

```python
from aind_smartspim_destripe.destriper import batch_filter

batch_filter(
    input_path="/path/to/images",
    output_path="/path/to/output",
    workers=8,
    chunks=1,
    high_int_filter_params={"wavelet": "db3", "level": None, "sigma": 64, "max_threshold": 3},
    low_int_filter_params={"wavelet": "db3", "level": None, "sigma": 128, "max_threshold": 12},
)
```

### Programmatic — single image

```python
from aind_smartspim_destripe.destriper import read_filter_save

read_filter_save(
    output_dir="/tmp",
    input_path="/path/to/image.tif",
    output_path="/path/to/output.tif",
    high_int_filter_params={"wavelet": "db3", "level": None, "sigma": 64, "max_threshold": 3},
    low_int_filter_params={"wavelet": "db3", "level": None, "sigma": 128, "max_threshold": 12},
)
```

---

## Key Parameters

| Parameter | Description | No-cells default | Cells default |
|-----------|-------------|-----------------|---------------|
| `wavelet` | PyWavelets wavelet family | `"db3"` | `"db3"` |
| `level` | Wavelet decomposition depth (`None` = full) | `None` | `None` |
| `sigma` | Gaussian filter sigma in frequency domain | `128` | `64` |
| `max_threshold` | Otsu multiplier for stripe mask | `12` | `3` |
| `retrospective` | Use BaSiC-estimated flatfields vs. microscope flats | `True` | `True` |

---

## Output

- **Format**: Zarr (uint16), Blosc zstd compression (level 3), chunk shape `(1, 1, 64, 128, 128)` in `(T, C, Z, Y, X)` order.
- **Multiscales**: 3 downsampled levels (factor 2 per spatial axis, windowed mean reduction).
- **Metadata**: OME-Zarr `.zattrs` with multiscale metadata; `image_destriping_<channel>_processing.json` with software version, parameters, and timestamps.

---

## Results

**Raw input**

![raw data](https://github.com/AllenNeuralDynamics/aind-smartspim-destripe/blob/main/metadata/imgs/raw.png?raw=true)

**Dual-band filtering** — stripes are suppressed but ringing artifacts appear around bright cells.

![dual band filtering](https://github.com/AllenNeuralDynamics/aind-smartspim-destripe/blob/main/metadata/imgs/filtered_dual_band.png?raw=true)

**Log-space FFT filtering** — stripes removed with minimal distortion of cell signal.

![log space filtering](https://github.com/AllenNeuralDynamics/aind-smartspim-destripe/blob/main/metadata/imgs/filtered_log_space.png?raw=true)

---

## Contributing

### Linters and testing

```bash
# Run tests with coverage report
coverage run -m unittest discover && coverage report

# Check docstring coverage
interrogate .

# Check code style
flake8 .

# Auto-format
black .

# Sort imports
isort .
```

### Pull requests

For internal members, please create a branch. For external members, please fork the repository and open a pull request from the fork. Commit messages follow [Angular](https://github.com/angular/angular/blob/main/CONTRIBUTING.md#commit) style:

```
<type>(<scope>): <short summary>
```

Types: `build`, `ci`, `docs`, `feat`, `fix`, `perf`, `refactor`, `test`

### Documentation

```bash
sphinx-apidoc -o doc_template/source/ src
sphinx-build -b html doc_template/source/ doc_template/build/html
```

See the [Sphinx installation guide](https://www.sphinx-doc.org/en/master/usage/installation.html) for setup instructions.
