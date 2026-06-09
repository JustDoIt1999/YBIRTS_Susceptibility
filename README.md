# Incorporating Temporal Order of Initiations Improves Retrogressive Thaw Slump Susceptibility Mapping on the Qinghai-Tibet Plateau

This repository provides the open-source code and execution guidelines for the academic paper **"Incorporating Temporal Order of Initiations Improves Retrogressive Thaw Slump Susceptibility Mapping on the Qinghai-Tibet Plateau"**.

This project proposes a novel retrogressive thaw slump (RTS) susceptibility mapping framework based on **weakly supervised ranking**. By defining **"Years Before Initiation" (YBI)** for each RTS-related sample to represent the physically observed temporal distance to initiation, the framework integrates **pairwise ranking constraints** within local spatial domains as an auxiliary supervision signal via the **NDCG-Aware LambdaRank Loss**. This approach significantly mitigates the limitations of conventional susceptibility models that typically compress multi-year inventories into static binary labels, discarding valuable temporal information on the evolution of slopes prior to initiation. In independent temporal validation (new RTS initiations in 2021–2022, n = 199), the framework substantially outperforms static and conventional dynamic binary models while maintaining a compact high-susceptibility zone (capturing 89.45% of new RTS initiations within the top 10% susceptibility zone, compared with 79.90% and 63.82% for the annual binary and static models, respectively).

##  Installation and Environment Setup

This project depends on spatial data processing and deep learning tools, including PyTorch, Google Earth Engine API, geemap, and Rasterio. **We strongly recommend using a Conda virtual environment for isolation**:

### 1. Create and Activate the Conda Environment
```bash
# Create the virtual environment (Python 3.10 is recommended)
conda create -n RTSSM python=3.10 -y

# Activate the environment
conda activate RTSSM
```

### 2. Install Project Dependencies
In the activated `RTSSM` environment, run the following command in the project root directory:
```bash
pip install -r requirements.txt
```

### 3. Google Earth Engine Authentication and Project Setup
If you need to perform data processing and spatial mapping, please complete the Google account authentication in the activated environment:
```bash
earthengine authenticate
```

Next, configure your Earth Engine Cloud Project ID. This project uses a `.env` file for configuration management to avoid modifying source code or manually setting system environment variables:
1. In the project root directory, copy the template to create a local configuration file:
   ```bash
   cp .env.example .env
   ```
2. Edit the generated `.env` file and replace `your-gcp-project-id` with the Google Cloud Project ID associated with your Earth Engine account:
   ```env
   EE_PROJECT_ID=your-gcp-project-id
   ```

---

## Three-Stage Workflow Usage Instructions

The complete workflow for data preparation, model training, and spatial projection is divided into three sequential steps:

```mermaid
graph LR
    Step01[1. Data Processing & Sampling] --> Step02[2. Model Training & Tuning] --> Step03[3. Annual Susceptibility Mapping]
```

---

### Step 1: Data Processing and Sampling (Step 01)
This step handles the processing of retrogressive thaw slump (RTS) samples, reconstruction of the pre-initiation YBI sequence, random extraction of pseudo-absence samples under spatial distance constraints, and extraction of 14 static and annual dynamic environmental predictor features using GEE.

#### 1. Core Module Responsibilities
The core logic for Step 1 is implemented in the sub-files under `Dataset_Construction/src/dataset_construction/`:
* [pa_sampling.py](file:///Users/lipengfei/Documents/RTS_susceptibility_v4/Dataset_Construction/src/dataset_construction/pa_sampling.py): Implements spatial distance constraints (distance to presence points $\ge 1000\text{ m}$, distance between points $\ge 1\text{ m}$), and performs random pseudo-absence sample extraction matching positive sample counts annually.
* [sample_tables.py](file:///Users/lipengfei/Documents/RTS_susceptibility_v4/Dataset_Construction/src/dataset_construction/sample_tables.py): Aggregates presence and pseudo-absence samples to generate base sample tables.
* [environmental_factors.py](file:///Users/lipengfei/Documents/RTS_susceptibility_v4/Dataset_Construction/src/dataset_construction/environmental_factors.py): Performs GEE predictor feature extraction. **When running this script, the program automatically detects and builds the required asset directories recursively within your GEE Asset library (`projects/your-project-id/assets`), with no need to create them manually on the Earth Engine Console**.
* [build_total_features.py](file:///Users/lipengfei/Documents/RTS_susceptibility_v4/Dataset_Construction/src/dataset_construction/build_total_features.py): Integrates the features exported from GEE with the sample tables, outputting the final training table `Total_features_sample.csv`.

#### 2. Execution Commands
Run the cross-platform pipeline script in the project root directory:
```bash
python Dataset_Construction/scripts/workflow/run_full_pipeline.py
```
> [!TIP]
> GEE feature extraction can be time-consuming. If you already have the extracted training table, you can rename it to `Total_features_sample.csv`, place it in the `Dataset_Construction/outputs/03_environmental_data/` directory, and skip directly to Step 2.

---

### Step 2: YBI Ranking Framework Model Training and Optimization (Step 02)
This step performs 5-fold cross-validation fold splitting, grid search for the Lambda ranking parameter, and training and saving of the optimal model weights.

#### 1. Execution Flow Description
The pipeline script sequentially triggers the following three core commands:
1. **`make-folds`**: Reads the spatial model routing configuration files (under `configs/routes/`, such as `dbscan.yaml`, `basin.yaml`, or `no_cluster.yaml`) and creates the 5-fold data partition.
2. **`sweep-lambda`**: Performs a grid search to find the optimal ranking loss weight parameter $\lambda_{rank}$ that balances classification and ranking objectives.
3. **`select-lambda --train`**: Reads the search logs, selects the best $\lambda_{rank}$, and trains/saves the 5-fold ensemble PyTorch model weights (`.pth`) and normalization Scalers (`.pkl`).

#### 2. Command Line Interface (CLI) Overrides
The pipeline script supports overriding parameters defined in the YAML files directly via CLI flags (convenient for testing and comparative experiments):
* Override batch size: `--batch-size N` (e.g., `--batch-size 256`)
* Override ranking loss mode: `--rank-loss-mode MODE` (e.g., `--rank-loss-mode ndcg_aware_lambdarank` or `disabled`)

#### 3. Execution Commands
```bash
cd YBI_Ranking_Framework
python scripts/workflow/run_full_pipeline.py
```
*(Note: The `run_full_pipeline.sh` script under scripts is provided for Linux server cluster scheduling.)*

---

### Step 3: Annual Susceptibility Mapping (Step 03)
Loads the 5-fold ensemble model and Scalers saved in Step 2, and performs tiled prediction mapping across large-scale annual study areas using GEE.

#### 1. Preparation: Manually Editing the Configuration File
Before running the projection, edit the configuration file [annual_projection.yaml](file:///Users/lipengfei/Documents/RTS_susceptibility_v4/Susceptibility_Mapping/configs/cvrank/annual_projection.yaml) to customize the prediction task:
* `route` / `batch_size` / `selection_strategy`: Controls how the program automatically locates the optimal model outputs from Step 2.
* `years`: Specifies the target years for mapping (e.g., 2016-2022).
* `download_workers`: The number of concurrent threads used to download data.

#### 2. Advanced Features of the Mapping Engine
* **Checkpoint Resume**: The mapping engine automatically writes progress manifests to the cache root. If the program or network is interrupted, running it again will resume from the checkpoint, avoiding redundant downloads and calculations.
* **Concurrency & Adaptive Fallback**: Uses asynchronous multi-threading to download feature tiles (default size: $512 \times 512$ pixels). If a tile fails to download due to high remote GEE server load, the engine automatically falls back to a smaller tile size ($256 \times 256$ pixels) and retries.
* **Bivariate Confidence Mapping**: In addition to exporting the **mean** of the 5-fold predictions as the susceptibility map, the engine automatically calculates the **standard deviation** as a spatial uncertainty metric. It then generates a $5 \times 5$ confidence class level map using the natural breaks classification method.

#### 3. Execution Commands
```bash
cd Susceptibility_Mapping
python annual_projection.py --config configs/cvrank/annual_projection.yaml --run
```
