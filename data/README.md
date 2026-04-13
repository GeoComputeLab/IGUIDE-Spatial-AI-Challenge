# Dataset Documentation: MURD-ViT

This directory contains the spatiotemporal dataset used for **MURD-ViT**, a multimodal deep learning pipeline for detecting urban retrofitting in Mecklenburg County (Charlotte), North Carolina.

## Dataset Overview

The dataset integrates visual and demographic layers from **2013 to 2023** to capture micro-scale urban transformations.

- **Study Area:** Mecklenburg County, North Carolina (Charlotte).
- **Temporal Scope:** 2013–2023.
- **Total Image Pairs:** 7,376 (Full Dataset).

## Directory Structure

```text
data/
├── Full_data/                # Complete dataset
│   ├── labels.xlsx           # Master annotation file
│   └── Streetview_data/      # Google Street View (GSV) image pairs
├── Sample_data/              # Subset of data for testing/quick runs
│   ├── labels.xlsx           # Annotations for sample data
│   └── Streetview_data/      # Corresponding GSV images
├── Shapefile/                # Census Block Group boundaries for Charlotte
│   └── mecklengburg_county_census_block_group_2013.*
├── figures/                  # Visualization figures
└── figures_low/              # Low-resolution figures for documentation
```

## Data Variables and Annotation (labels.xlsx)

The `labels.xlsx` file serves as the primary metadata source for loading image pairs and demographic features.

| Column                   | Description                                           |
| :----------------------- | :---------------------------------------------------- |
| `ID`                     | Census Block Group ID (GEOID code)                    |
| `Latitude`               | Latitude coordinate of the GSV location               |
| `Longitude`              | Longitude coordinate of the GSV location              |
| `Heading`                | Direction of the camera (0, 90, 180, 270)             |
| `Start`                  | Date of the "Before" image (YYYY-MM-DD)               |
| `End`                    | Date of the "After" image (YYYY-MM-DD)                |
| `Change`                 | Urban retrofitting category (Encoded label)           |
| `PopDen_Diff`            | Population density change between Start and End dates |
| `PopDen_Diff_Percentage` | Percentage change in population density               |

## Retrofitting Categories

The dataset classifies urban changes into five categories:

1. **No Change (0):** Baseline/Static urban environments.
2. **Re-building (1):** Structural optimizations and densification.
3. **Re-inhabitation (2):** Housing quality and socioeconomic upgrades.
4. **Re-transportation (3):** Sustainable mobility and network changes.
5. **Re-capital (4):** Economic and commercial development.

## Image Organization

Google Street View (GSV) images are organized hierarchically:
`Streetview_data / {CensusBlockGroupID} / {Lat}_{Lon}_{Heading} / {Month Year}.png`

Example:
`data/Sample_data/Streetview_data/371190001001/35.21987836403704_-80.85295426821791_0/May 2014.png`

## Usage for Training

- **Visual Features:** Start and End image pairs are used as inputs to the Vision Transformer (ViT).
- **Demographic Features:** `PopDen_Diff` and `PopDen_Diff_Percentage` are used as socio-economic context for the model.
- **Spatial Stratified Sampling:** K-Means clustering is applied to `Latitude` and `Longitude` to ensure geographic diversity in training and validation splits.

## Data Loading and Subset Generation

The process of moving from raw Excel annotations to model-ready `DataLoader` objects follows a specific sequence of initialization and subsetting:

### 1. Dataset Initialization

The `UrbanRetrofittingDataset` is first initialized for the entire dataset. In the workflow, it is initialized **twice** to allow for different image transformations (augmentation for training vs. static resizing for validation):

```python
# Initializing the dataset with training and validation transforms
train_dataset_base = UrbanRetrofittingDataset(excel_File, image_dir, train_transform)
val_dataset_base = UrbanRetrofittingDataset(excel_File, image_dir, val_transform)
```

### 2. Spatial Stratified Splitting

To ensure geographic and class balance, a custom `spatial_stratified_split` function is used. This function:

- Performs **K-Means Clustering** on the `Latitude` and `Longitude` of the full dataset.
- Randomly samples 20% of data from each cluster and each class for validation.
- Returns the specific indices for the split.

```python
# Generating indices for training and validation subsets
train_indices, val_indices, _ = spatial_stratified_split(
    dataset=train_dataset_base,
    val_ratio=0.2,
    n_clusters=NO_OF_CLUSTERS
)
```

### 3. Creating Subsets

Using the generated indices, PyTorch `Subset` objects are created. These subsets act as views into the base dataset, ensuring that the training subset uses the `train_transform` and the validation subset uses the `val_transform`.

```python
# Creating the final training and validation subsets
train_subset = Subset(train_dataset_base, train_indices)
val_subset = Subset(val_dataset_base, val_indices)
```

### 4. DataLoader Delivery

Finally, these subsets are passed into the `DataLoader` to handle batching, shuffling, and multi-threaded data loading.

```python
# Creating DataLoaders for model consumption
train_loader = DataLoader(
    train_subset,
    batch_size=BATCH_SIZE,
    sampler=WeightedRandomSampler(...), # Optional for class balance
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_subset,
    batch_size=BATCH_SIZE,
    shuffle=False
)
```

## [User specifically interested in using data directly rather than from excel file annotations can follow here](./../notebook.ipynb#direct-use)

## Data Specifications & Original Structure

Understanding the raw data structure is critical for reproducibility. The original annotations in `labels.xlsx` contain both spatial identifiers and temporal metadata.

### Raw Data Types & Dimensions

| Field                      | Data Type (Raw) | Model Input Type    | Dimensions/Shape  |
| :------------------------- | :-------------- | :------------------ | :---------------- |
| **GSV Images**             | `.png` (RGB)    | `torch.Tensor`      | `(3, 2420, 1326)` |
| **Latitude/Longitude**     | `np.longdouble` | Metadata/Clustering | N/A               |
| **Heading**                | `int`           | Pathing Metadata    | N/A               |
| **Start/End Dates**        | `datetime`      | Pathing Metadata    | N/A               |
| **ID (Census ID)**         | `int`/`str`     | Pathing Metadata    | N/A               |
| **PopDen_Diff**            | `float`         | `torch.Tensor`      | `(1,)`            |
| **PopDen_Diff_Percentage** | `float`         | `torch.Tensor`      | `(1,)`            |
| **Change (Label)**         | `str`           | `torch.Tensor`      | `(1,)` (Encoded)  |

### Variables Not Included as Model Inputs

While present in the dataset for organization and analysis, the following variables are **not** passed into the neural network layers:

- **`ID`**: Used only to locate the subfolder within `Streetview_data`.
- **`Latitude` / `Longitude`**: Used for spatial stratified sampling (K-Means) during the split phase, but not as direct features.
- **`Heading`**: Used to identify the specific orientation folder.
- **`Start` / `End`**: Used to identify the specific temporal image files.

## Data Loading Pipeline

The `UrbanRetrofittingDataset` class (inherited from `torch.utils.data.Dataset`) handles the transition from raw files to model-ready tensors.

1.  **Coordinate Precision**: Latitude and Longitude are loaded as `np.longdouble` to maintain sub-meter precision required for matching street-view panorama locations.
2.  **Temporal Pairing**: For a given row, the loader fetches two images: one corresponding to the `Start` date and one to the `End` date from the hierarchical folder structure.
3.  **Transformations**:
    - Images are resized to `224x224`.
    - Normalized using image statistics (`mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]`).
    - Training data undergoes augmentation (Random Horizontal Flip, Rotation, Color Jitter).

## How Data Feeds into the MURD-ViT Model

The model is a **multimodal fushion architecture**. The `DataLoader` provides batches of 4-tuples: `(start_image, end_image, socio_economic_data, label)`.

### Input Flow into Model Layers:

1.  **Visual Branch (ViT Base)**:
    - `start_image` ($3 \times 224 \times 224$) $\rightarrow$ ViT Encoder $\rightarrow$ `start_feat` ($768$ dim)
    - `end_image` ($3 \times 224 \times 224$) $\rightarrow$ ViT Encoder $\rightarrow$ `end_feat` ($768$ dim)
    - **Temporal Difference**: The model explicitly calculates `diff_feat = end_feat - start_feat`.
2.  **Demographic Branch (MLP)**:
    - `socio_economic_data` ($2$ dim: `PopDen_Diff`, `PopDen_Diff_Percentage`) $\rightarrow$ 2-layer MLP $\rightarrow$ `numeric_feat` ($32$ dim)
3.  **Feature Fusion**:
    - The features are concatenated: `[start_feat, end_feat, diff_feat, numeric_feat]`.
    - **Combined Feature Dimension**: $768 + 768 + 768 + 32 = 2336$.
4.  **Classification Head**:
    - The $2336$-dim vector passes through a final MLP (Linear $\rightarrow$ ReLU $\rightarrow$ Dropout $\rightarrow$ Linear) to output the probability distribution for the 5 retrofitting categories.

## Data Sources

- **Street View Imagery:** Retrieved using GSV Panorama IDs and the Street View Static API.
- **Demographic Data:** Sourced from the [U.S. Census Bureau Data Portal](https://data.census.gov/).
