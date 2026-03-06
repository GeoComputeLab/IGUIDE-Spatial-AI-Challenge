# MURD-ViT: Urban Retrofitting Detection with Vision Transformer

## 🎓 Citation

If you need to use this work in your research, its paper is under revision. So, contact authors.

---

![MURD-ViT Architecture](data/figures_low/architecture.png)

**MURD-ViT** (Multimodal Urban Retrofitting Detection - Vision Transformer) is a deep learning pipeline designed to detect urban retrofitting interventions—micro-scale upgrades to existing urban spaces—using temporal Google Street View (GSV) imagery and demographic data.

By integrating Vision Transformer (ViT) embeddings from "before" and "after" image pairs with socio-economic indicators (population density changes), MURD-ViT classifies urban changes into five distinct categories: **No Change**, **Re-building**, **Re-inhabitation**, **Re-transportation**, and **Re-capital**.

---

## 🚀 Key Features

- **Multimodal Fusion**: Combines temporal image pairs with demographic features (population density and percentage change).
- **ViT Backbone**: Leverages Vision Transformers to capture global spatial dependencies in street view images.
- **Spatial Stratified Sampling**: Implements K-Means clustering to ensure geographic diversity and balanced class distribution across training/validation splits.
- **Robust Evaluation**: Uses Top-2 Accuracy to address the inherent rarity and class imbalance of urban retrofitting events.
- **Geospatial Visualization**: Includes interactive Folium maps to visualize training and validation sample distributions across study areas.

---

## 🏗️ Model Architecture

The MURD-ViT architecture consists of:

1.  **Image Encoder**: A pre-trained `vit_base_patch16_224` that extracts high-dimensional embeddings from "Before" and "After" images.
2.  **Temporal Difference Layer**: Computes the feature-level difference between the two image states.
3.  **Demographic MLP**: A multi-layer perceptron that encodes population density metrics.
4.  **Classification Head**: A fused layer that integrates image features, their differences, and demographic embeddings to predict the retrofit category.

---

## 📊 Dataset

The project uses data from **Mecklenburg County, North Carolina** (Charlotte):

- **Images**: 7,376 temporal pairs of Google Street View images.
- **Demographics**: Population counts and density changes (2013–2023) from the US Census Bureau.
- **Classes**:
  - `No Change`: Baseline (majority class)
  - `re-building`: Physical structure optimizations
  - `re-inhabitation`: Housing/socio-economic upgrades
  - `re-transportation`: Mobility network improvements
  - `re-capital`: Sustainable economic development

---

## 🛠️ Installation

Ensure you have Python 3.9+ installed. You can install the dependencies via `pip`:

```bash
pip install -r requirements.txt
```

**Core Dependencies:**

- `torch`, `torchvision`, `timm` (Deep Learning)
- `pandas`, `geopandas`, `numpy` (Data Processing)
- `opencv-python`, `Pillow` (Computer Vision)
- `scikit-learn` (ML Utilities)
- `folium`, `matplotlib`, `seaborn` (Visualization)

---

## 📖 Usage

The main entry point is the `notebook.ipynb` file, which walks through the entire pipeline:

1.  **Setup**: Configure device (CPU/CUDA) and hyperparameters.
2.  **Data Loading**: Uses `UrbanRetrofittingDataset` to pair images and census data.
3.  **Spatial Split**: Runs K-Means clustering on coordinates for stratified sampling.
4.  **Training**: Run the `train_epoch` loop. A `RUN_SAMPLE` flag is provided to quickly test the pipeline on a small subset of data.
5.  **Evaluation**: Validates performance using Top-2 Accuracy and generates confusion matrices.

### Configuration

In the notebook, toggle `RUN_SAMPLE = True` to use the sample dataset provided in `data/Sample_data`. Set it to `False` for full-scale training.

---

## 📁 Project Structure

```text
.
├── data/
│   ├── figures_low/         # Architectural and result visualizations
│   ├── figures/             # High resolution images of figures_low
│   ├── Full_data/           # Full dataset (labels and images)
│   ├── Sample_data/         # Small subset for quick testing
│   └── Shapefile/           # Census block group boundaries
├── logs/                    # Training and validation logs
├── Model_Checkpoints/       # Saved .pth weights
├── notebook.ipynb           # Main implementation notebook
└── requirements.txt         # Project dependencies
```

---

## 📈 Results

The model is evaluated based on its ability to identify "Where, When, and What" retrofitting occurred. Due to the high imbalance (most locations experience "No Change"), performance is tracked via per-class accuracy and Top-2 accuracy.

![Model Prediction](data/figures_low/model_prediction.png)

---

## 📝 License

This project is licensed under the [MIT License](LICENSE).
