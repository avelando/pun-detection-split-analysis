# Portuguese Pun Detection

This repository contains the source code, corpora, and experimental materials used to reproduce experiments for automatic pun detection in Portuguese.

The task is formulated as a binary classification problem:

- `0`: non-pun text;
- `1`: pun text.

The main objective is to evaluate how different corpus organizations affect the performance of pun detection models. Two experimental scenarios are considered:

1. **Original corpus**: the original train, validation, and test splits.
2. **Reorganized corpus**: a split reorganized by base identifier, ensuring that related pun and non-pun versions remain in the same partition.

Each model was executed with the corresponding corpus version in order to reproduce the reported results. Therefore, the BERTimbau-based model and the ensemble model were both evaluated on the original corpus and on the reorganized corpus.

---

## Repository Structure

```text
.
├── data/
│   ├── original/
│   │   ├── train.jsonl
│   │   ├── validation.jsonl
│   │   └── test.jsonl
│   │
│   └── reorganized/
│       ├── train.jsonl
│       ├── validation.jsonl
│       └── test.jsonl
│
├── src/
│   ├── data/
│   │   ├── make_reorganized_split.py
│   │   └── make_reorganized_split.ipynb
│   │
│   └── models/
│       ├── train_bertimbau.py
│       ├── train_bertimbau.ipynb
│       ├── train_ensemble.py
│       └── train_ensemble.ipynb
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Data

The `data/` directory contains the corpus files used in the experiments.

### Original corpus

The original split is stored in:

```text
data/original/
```

It contains:

```text
train.jsonl
validation.jsonl
test.jsonl
```

This version corresponds to the original experimental scenario.

### Reorganized corpus

The reorganized split is stored in:

```text
data/reorganized/
```

It contains:

```text
train.jsonl
validation.jsonl
test.jsonl
```

In this version, examples related by the same base identifier are kept in the same partition. This prevents highly similar pun and non-pun versions from being distributed across different splits, such as training and test sets.

---

## Source Code

The source code is organized under the `src/` directory.

---

### `src/data/make_reorganized_split.py`

This script is responsible for generating the reorganized version of the corpus.

It performs the following steps:

- reads the original `train.jsonl`, `validation.jsonl`, and `test.jsonl` files;
- extracts the base identifier from each example;
- groups related pun and non-pun examples by their base identifier;
- redistributes the grouped pairs into train, validation, and test partitions;
- preserves proportions compatible with the original corpus organization;
- ensures that related examples are not split across different partitions;
- writes the reorganized files to `data/reorganized/`.

A notebook version is also available:

```text
src/data/make_reorganized_split.ipynb
```

The notebook is provided for inspection and interactive execution.

---

### `src/models/train_bertimbau.py`

This script trains and evaluates a BERTimbau-based classifier for binary pun detection.

It performs the following steps:

- reads the corpus files from the expected input directory;
- loads the training, validation, and test sets;
- tokenizes the texts;
- trains a neural classifier based on BERTimbau;
- evaluates the model on the test set;
- reports accuracy, precision, recall, and F1-score;
- displays the confusion matrix.

A notebook version is also available:

```text
src/models/train_bertimbau.ipynb
```

The notebook version can be used for interactive analysis or execution in environments such as Jupyter or Google Colab.

---

### `src/models/train_ensemble.py`

This script trains and evaluates an ensemble of traditional machine learning classifiers.

It performs the following steps:

- reads the corpus files from the expected input directory;
- converts the texts into TF-IDF representations;
- trains traditional supervised classifiers;
- combines the classifiers using an ensemble strategy;
- evaluates the final model on the test set;
- reports classification metrics.

A notebook version is also available:

```text
src/models/train_ensemble.ipynb
```

---

## Installation

It is recommended to create a virtual environment before installing the dependencies.

### Windows PowerShell

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### Linux/macOS

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Dependencies

The project dependencies are listed in:

```text
requirements.txt
```

Install them with:

```bash
pip install -r requirements.txt
```

A compatible `requirements.txt` should include the main libraries used by the scripts, such as:

```text
numpy
pandas
scikit-learn
scipy
nltk
torch
transformers
tqdm
jupyter
ipykernel
```

For GPU-based execution of BERTimbau, make sure that the installed PyTorch version is compatible with the local CUDA version.

---

## Data Format

The corpus files use the JSONL format. Each line corresponds to one independent JSON object.

A typical example follows this structure:

```json
{
  "id": "1.3.H",
  "text": "Example text",
  "label": 1
}
```

### Expected fields

| Field | Description |
|---|---|
| `id` | Example identifier |
| `text` | Text to be classified |
| `label` | Binary label: `0` for non-pun and `1` for pun |

---

## Reproducing the Experiments

The training scripts expect the corpus files to be available in a directory named `corpus/`, with the following structure:

```text
corpus/
├── train.jsonl
├── validation.jsonl
└── test.jsonl
```

Before running each experiment, copy the desired corpus version into the `corpus/` directory.

---

# Experiments with the Original Corpus

## 1. Prepare the original corpus

### Windows PowerShell

```powershell
Remove-Item -Recurse -Force corpus -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Path corpus

Copy-Item data\original\train.jsonl corpus\train.jsonl
Copy-Item data\original\validation.jsonl corpus\validation.jsonl
Copy-Item data\original\test.jsonl corpus\test.jsonl
```

### Linux/macOS

```bash
rm -rf corpus
mkdir corpus

cp data/original/train.jsonl corpus/train.jsonl
cp data/original/validation.jsonl corpus/validation.jsonl
cp data/original/test.jsonl corpus/test.jsonl
```

---

## 2. Run BERTimbau on the original corpus

```bash
python src/models/train_bertimbau.py
```

---

## 3. Run the ensemble on the original corpus

```bash
python src/models/train_ensemble.py
```

---

# Experiments with the Reorganized Corpus

## 1. Prepare the reorganized corpus

### Windows PowerShell

```powershell
Remove-Item -Recurse -Force corpus -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Path corpus

Copy-Item data\reorganized\train.jsonl corpus\train.jsonl
Copy-Item data\reorganized\validation.jsonl corpus\validation.jsonl
Copy-Item data\reorganized\test.jsonl corpus\test.jsonl
```

### Linux/macOS

```bash
rm -rf corpus
mkdir corpus

cp data/reorganized/train.jsonl corpus/train.jsonl
cp data/reorganized/validation.jsonl corpus/validation.jsonl
cp data/reorganized/test.jsonl corpus/test.jsonl
```

---

## 2. Run BERTimbau on the reorganized corpus

```bash
python src/models/train_bertimbau.py
```

---

## 3. Run the ensemble on the reorganized corpus

```bash
python src/models/train_ensemble.py
```

---

## Expected Results

The experiments compare two models across two corpus configurations.

| Corpus | Model | Accuracy |
|---|---:|---:|
| Original | Ensemble | 0.80 |
| Original | BERTimbau | 0.68 |
| Reorganized | Ensemble | 0.46 |
| Reorganized | BERTimbau | 0.76 |

Small variations may occur depending on the execution environment, library versions, hardware, GPU availability, and random initialization.

---

## Recreating the Reorganized Corpus

The reorganized corpus is already available in:

```text
data/reorganized/
```

To recreate it, run:

```bash
python src/data/make_reorganized_split.py
```

The expected output is:

```text
data/reorganized/train.jsonl
data/reorganized/validation.jsonl
data/reorganized/test.jsonl
```

Before running the script, verify that the input and output paths are correctly configured in the source code.

---

## Notes on the Notebooks

Notebook files are provided as auxiliary material for interactive inspection and reproduction.

For automated or terminal-based reproduction, prefer using the `.py` scripts:

```text
src/data/make_reorganized_split.py
src/models/train_bertimbau.py
src/models/train_ensemble.py
```

The notebooks may be useful for:

- inspecting intermediate steps;
- validating corpus distributions;
- checking model outputs;
- adapting the experiments;
- running the workflow in notebook-based environments.

---

## Version Control

The `.gitignore` file is used to avoid tracking unnecessary files, temporary files, caches, virtual environments, and old project folders.

Examples of ignored files and directories include:

```text
venv/
.venv/
__pycache__/
*.pyc
.ipynb_checkpoints/
BERTimbau/
ensemble/
relatorio_splits/
main.py
```

This keeps the repository cleaner and avoids uploading unnecessary files to GitHub.

---

## Reproduction Summary

To reproduce the experiments:

1. Create and activate a virtual environment.
2. Install the dependencies from `requirements.txt`.
3. Copy the desired corpus version into the `corpus/` directory.
4. Run the BERTimbau script.
5. Run the ensemble script.
6. Repeat the process for the other corpus version.
7. Compare the results obtained for each model and corpus scenario.

---

## Research Use

This repository is intended for research reproducibility and experimental documentation. The provided scripts and corpus organization allow the comparison of model performance under different split configurations for Portuguese pun detection.

---

## License

This repository is made available for research and experimental reproduction purposes.