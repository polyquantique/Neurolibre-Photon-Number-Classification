# Neurolibre-Photon-Number-Classification
Neurolibre format for `Nonlinear Dimensionality Reduction for Enhanced Unsupervised Classification in Transition Edge Sensors`


## Build Process (TODO)

### content

- `_config.yml` : Configuration (reference [here](https://jupyterbook.org/en/stable/customize/config.html))
    - Used default
- `_toc.yml` : File structure (reference [here](https://jupyterbook.org/en/stable/structure/toc.html))

### binder

- `data_requirement.json` : Template for Zenodo
- `requirements.txt` : Necessary packages
- `runtime.txt` : Python-3.12

### draft pdf

- `draft-pdf.yml` : Github now requires v4 instead of `actions/checkout@v3`

    ```
    jobs:
    paper:
        runs-on: ubuntu-latest
        name: Paper Draft
        steps:
        - name: Checkout
            uses: actions/checkout@v4
    ```
- `draft-pdf.yml` :  Github now requires v1 instead of `actions/upload-artifact@v2`

    ```
    - name: Upload
        uses: actions/upload-artifact@v2
    ```