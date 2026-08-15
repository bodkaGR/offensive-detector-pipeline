# Pipeline for training Offensive Detection Model

Training pipeline for an offensive text detection model. It fine-tunes an MPNet-based sentence encoder using an SBERT-style methodology and produces a model artifact that is published to the Hugging Face Hub for consumption by [`offensive-detector-api`](https://github.com/bodkaGR/offensive-detector-api).

## Overview

The core model is built on `all-mpnet-base-v2`, extended with a custom architecture:

- **Transformer encoder blocks** on top of the MPNet backbone
- **Attention pooling** over token representations
- **Classification head** for the final offensive / non-offensive decision

The best-performing configuration (MPNet + custom TransformerEncoder with Attention Pooling) reaches:

| Metric      | Score  |
|-------------|--------|
| Accuracy    | 0.9521 |
| F1 weighted | 0.9516 |
| F1 macro    | 0.9126 |
| F1 binary   | 0.9713 |
| ROC-AUC     | 0.9831 |
| PR-AUC      | 0.9963 |

on the [*Hate Speech and Offensive Language*](https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset/data) dataset, outperforming an MPNet + MLP baseline.

## Project structure

```
.
├── data/               # Raw/processed datasets and data artifacts
├── scripts/            # Entry-point scripts (training, prediction)
├── src/                # Pipeline source code (model, training, data processing, configuration)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

## Tech stack

- **PyTorch** / **Transformers** / **Sentence-Transformers** (`tokenizers`, `huggingface_hub`) for the model itself
- **MLflow** for experiment tracking
- **scikit-learn**, **pandas**, **numpy** for data processing and evaluation
- **NLTK** for text preprocessing
- **matplotlib** / **seaborn** for result visualization
- **Docker** / **docker-compose** for reproducible execution

See `requirements.txt` for exact pinned versions.

## Getting started

### 1. Clone and install

```bash
git clone https://github.com/bodkaGR/offensive-detector-pipeline.git
cd offensive-detector-pipeline

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configure environment

Copy the example env file and fill in your Hugging Face credentials:

```bash
cp .env.example .env
```

```
# Hugging Face Hub
HF_REPO_ID=bodka-gr/offensive_text_detector
HF_TOKEN=<your-access-token>
```

`HF_TOKEN` needs write access if you intend to push a newly trained model to the Hub.

### 3. Run the pipeline

Training/evaluation entry points live under `scripts/`. A typical run looks like:

```bash
python scripts/train.py --epochs 10 --push_to_hub
```

#### Parameters

**Model**
 
| Parameter        | Type | Default | Description                                                                                                                                   |
|------------------|---|---|-----------------------------------------------------------------------------------------------------------------------------------------------|
| `--data_path`    | `str` | `data/labeled_data.csv` | Path to the labeled training data (CSV).                                                                                                      |
| `--epochs`       | `int` | `10` | Number of training epochs.                                                                                                                    |
| `--freeze_sbert` | flag | `False` | Freezes the MPNet backbone weights, training only the added layers (TransformerEncoder, attention pooling, classification head) on top of it. |
| `--accum_steps`  | `int` | `1` | Number of gradient accumulation steps - effectively multiplies the batch size without increasing memory usage.                                |
 
**MLflow**
 
| Parameter | Type | Default | Description |
|---|---|---|---|
| `--experiment` | `str` | `""` | Name of the MLflow experiment to log the run under. Defaults to MLflow's default experiment if left empty. |
| `--run_name` | `str` | `""` | Name for this specific MLflow run, useful for identifying it in the MLflow UI. |
 
**Hugging Face Hub**
 
| Parameter | Type | Default | Description |
|---|---|---|---|
| `--push_to_hub` | flag | `False` | Publishes the trained model and tokenizer to the Hugging Face Hub after training. |
| `--hf_repo_id` | `str` | `""` | Target Hub repo ID to push to (e.g. `bodka-gr/offensive_text_detector`). Falls back to `HF_REPO_ID` from `.env` if not provided. |
| `--hf_public` | flag | `False` | Makes the pushed Hub repo public. Omit to keep it private. |
| `--hf_commit_message` | `str` | `""` | Custom commit message for the Hub push. |

### 4. Docker (optional)

For a reproducible environment:

```bash
docker compose up --build
```

## Output

A successfully trained model is pushed to the Hugging Face Hub repository configured via `HF_REPO_ID` (`bodka-gr/offensive_text_detector` by default). The [`offensive-detector-api`](https://github.com/bodkaGR/offensive-detector-api) service loads the model directly from that Hub repo at runtime.

## Related repositories

- [`offensive-detector-api`](https://github.com/bodkaGR/offensive-detector-api) - FastAPI service that serves predictions from the model trained here.
