# Third-party components

The organizer-provided competition input and output Python interfaces retain their original documentation. Numerical signal processing and model code are included with the submitted source.

Runtime dependencies retain their distributed license terms: Python (PSF), PyTorch (BSD-style and bundled notices), NumPy/SciPy/pandas/h5py (BSD-style), imufusion (MIT), threadpoolctl (BSD-style), uv (MIT/Apache-2.0), and the transitive packages recorded in the dependency lock. The imufusion license is included under `licenses/`.

The Windows bootstrap downloads the uv release identified in `environment/bootstrap.json` and verifies its SHA256 before use. Python dependencies are installed from the versions and hashes in `environment/requirements.lock`.
