# Neurolibre-Photon-Number-Classification
Neurolibre format for `Nonlinear Dimensionality Reduction for Enhanced Unsupervised Classification in Transition Edge Sensors`


## Structure

### Pipeline

The repo is divided to be executed in stages :

- First Preprocessing (On external machine):
    - Run `Methods_Geometric.ipynb`, `Methods_Uniform.ipynb`, and `Methods_Large.ipynb`.
        - Compute `Preprocess/Low_Dimensional` (Done on computer with more than 32Gb of RAM)
        - Specify initial position in `Preprocess/Mean_Clusters`.
        - This step requires the data from Zenodo.
- Second Preprocessing (On Neurolibre with 12Gb of RAM):
    - Run `Methods_Geometric.ipynb`, `Methods_Uniform.ipynb`, and `Methods_Large.ipynb`
        - Since `Preprocess/Low_Dimensional` exists the file loads the data instead of computing the low-dimensional spaces.
        - This step requires the data from Zenodo.
        - In this step 
- Plot (On Neurolibre with 4 Gb of RAM)
    - Run `Results.ipynb`
        - Load preprocessed data saved in `Results`
        - We could add density plots from the `Methods_XXXX.ipynb` files in this step.
        - This step does not require the data from Zenodo.


