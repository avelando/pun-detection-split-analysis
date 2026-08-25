# Portuguese Pun Detection — When the Split Matters

This repository contains the source code, corpora, experimental artifacts, and
reproducibility materials for the study on how the **organization of the
train/validation/test splits** affects automatic **pun detection in Portuguese**.

The task is a binary classification problem:

- `0`: non-pun text;
- `1`: pun text.

The corpus is built from micro-edited pairs: every pun has a minimally edited
non-pun counterpart that shares the same **Base Identifier (IDB)**. The central
question of this study is what happens to model evaluation when the two versions
of a pair are allowed to fall into different partitions (training vs. test)
versus when they are forced to stay together. To isolate that effect, the
experiments compare split strategies at two extremes and run each one over
multiple random seeds, so the reported differences can be read against their
variance rather than a single lucky split.

Two models are evaluated under every configuration:

- a neural baseline based on **BERTimbau large** (`neuralmind/bert-large-portuguese-cased`);
- an **ensemble** of traditional classifiers (Random Forest + Logistic Regression + SVM over TF-IDF), following the lower-cost setting explored by Leal et al.

---

## Table of Contents

1. [Project Structure](#1-project-structure)
2. [Library Versions](#2-library-versions)
3. [Execution Environment (Hardware)](#3-execution-environment-hardware)
4. [Citing the Puntuguese Corpus](#4-citing-the-puntuguese-corpus)
5. [Citing This Work](#5-citing-this-work)
6. [Reproducing the Experiments](#6-reproducing-the-experiments)
7. [Data Format](#7-data-format)
8. [License](#8-license)

---

## 1. Project Structure

```text
.
├── data/                              # All corpora and split bookkeeping
│   ├── puns.json                      # Full raw source corpus (all pun/non-pun instances)
│   ├── split_comparison.csv           # Cross-split pair rate per strategy/seed (incl. original)
│   ├── split_summary.csv              # Per-strategy/seed split sizes and cross-split counts
│   │
│   ├── original/                      # Original Puntuguese split (as distributed)
│   │   ├── train.jsonl
│   │   ├── validation.jsonl
│   │   └── test.jsonl
│   │
│   ├── pair_controlled/               # "Clean" strategy: pairs never cross splits (cross-split rate = 0.0)
│   │   ├── summary.csv                # Summary across all seeds for this strategy
│   │   └── seed_<13|21|40|42|73|101>/ # One folder per random seed
│   │       ├── train.jsonl
│   │       ├── validation.jsonl
│   │       ├── test.jsonl
│   │       ├── metadata.json          # Strategy, seed, sizes, class distribution, cross-split stats
│   │       ├── pair_matrix.csv        # Split-by-split cross-tabulation of pun/non-pun versions
│   │       └── inspection.csv         # Per-example split assignment (auditing)
│   │
│   └── max_cross_split/               # "Leaky" strategy: maximizes pairs crossing splits (rate = 0.6)
│       ├── summary.csv
│       └── seed_<13|21|40|42|73|101>/ # Same file layout as pair_controlled
│
├── results/                          # Experimental outputs
│   ├── bertimbau/
│   │   ├── all_runs.csv               # Every run (all strategies/seeds) in one table
│   │   ├── confusion_counts_all.csv   # TP/TN/FP/FN for every run
│   │   ├── error_analysis_all.csv     # Punning-sign categories of false negatives, all runs
│   │   ├── summary_mean_std.csv       # Mean ± std per strategy across seeds
│   │   ├── original_result.csv        # Single-run result on the original corpus
│   │   ├── original/                  # Full artifacts for the original-corpus run
│   │   ├── pair_controlled/
│   │   │   ├── runs.csv               # Per-seed summary for this strategy
│   │   │   └── seed_<...>/            # Full artifacts for each seed (see below)
│   │   └── max_cross_split/
│   │       ├── runs.csv
│   │       └── seed_<...>/
│   │
│   └── ensemble/                      # Same layout as bertimbau/ (with saved models)
│       └── ...
│
├── src/
│   ├── data/
│   │   └── make_reorganized_split.ipynb   # Builds the pair_controlled / max_cross_split corpora
│   └── models/
│       ├── train_bertimbau.ipynb          # Trains + evaluates the BERTimbau baseline
│       └── train_ensemble.ipynb           # Trains + evaluates the traditional ensemble
│
├── .gitignore
├── requirements.txt                  # Pinned runtime dependencies
├── LICENSE                           # MIT license (source code)
├── LICENSE-DATA                      # CC-BY-SA-4.0 notice (corpus/data, inherited from Puntuguese)
└── README.md
```

### What each per-seed result folder contains

Every `results/<model>/<strategy>/seed_<n>/` folder holds the complete evidence
for one run, so any number in the paper can be traced back to its source:

| File | Description |
|---|---|
| `metrics.json` | Accuracy, per-class precision/recall/F1, macro/weighted averages |
| `classification_report.csv` | The same metrics in tabular form |
| `confusion_matrix.csv` | Full 2×2 confusion matrix |
| `confusion_counts.csv` | TP / TN / FP / FN counts |
| `error_analysis.csv` | Punning-sign categories among false negatives (none / homophone / homograph / both) |
| `error_analysis.png` | Chart of the error analysis above |
| `metadata.json` | Full run configuration (model, seeds, hyperparameters, hardware, library versions) |
| `training_history.csv` | Per-epoch training/validation curve **(BERTimbau only)** |
| `ensemble.joblib` | Serialized trained ensemble model **(ensemble only)** |

### The three split strategies at a glance

| Strategy | Cross-split pairs | Cross-split rate | Role in the study |
|---|---:|---:|---|
| `original` | 1306 | 0.458 | The corpus as originally distributed |
| `pair_controlled` | 0 | 0.000 | Clean evaluation — each pair stays in one split |
| `max_cross_split` | 1710 | 0.600 | Stress test — deliberately maximizes pairs crossing splits |

All strategies keep the same overall sizes (3990 train / 570 validation / 1140
test) and the same 50/50 class balance in every partition.

---

## 2. Library Versions

All experiments were run on **Python 3.14.4**. The pinned versions live in
[`requirements.txt`](requirements.txt); the table below groups the main
libraries by the stage that uses them.

### Split reorganization (`src/data/make_reorganized_split.ipynb`)

| Library | Version | Role |
|---|---|---|
| numpy | 2.5.2 | Shuffling / numeric operations |
| pandas | 3.0.5 | Reading, grouping and writing the split tables |

(Plus the Python standard library: `json`, `re`, `pathlib`.)

### BERTimbau training (`src/models/train_bertimbau.ipynb`)

| Library | Version | Role |
|---|---|---|
| torch | 2.13.0 (`+cu130`) | Neural training backend |
| transformers | 5.15.0 | BERTimbau model, tokenizer, `Trainer` |
| accelerate | 1.14.0 | Training loop / device placement backend for `Trainer` |
| scikit-learn | 1.9.0 | Evaluation metrics |
| numpy | 2.5.2 | Numeric operations |
| pandas | 3.0.5 | Result tables |
| matplotlib | 3.11.1 | Error-analysis charts |

### Ensemble training (`src/models/train_ensemble.ipynb`)

| Library | Version | Role |
|---|---|---|
| scikit-learn | 1.9.0 | TF-IDF, Random Forest, Logistic Regression, SVM, soft-voting ensemble, metrics |
| nltk | 3.10.3 | Portuguese stopword removal |
| joblib | 1.5.3 | Saving the trained ensemble (`ensemble.joblib`) |
| numpy | 2.5.2 | Numeric operations |
| pandas | 3.0.5 | Result tables |
| matplotlib | 3.11.1 | Error-analysis charts |

> Full pinned list (including `scipy==1.18.0`, `ipykernel==7.3.0`) is in
> `requirements.txt`. For GPU execution of BERTimbau, make sure the installed
> PyTorch build matches your local CUDA version.

---

## 3. Execution Environment (Hardware)

The experiments were executed on a single Linux workstation with the following
main components:

| Component | Value |
|---|---|
| Operating system | Ubuntu 26.04 LTS (Resolute Raccoon) |
| Kernel | 7.0.0-28-generic |
| CPU | Intel Core i5-10400F @ 2.90 GHz (6 cores / 12 threads) |
| RAM | 62 GB (64 GiB) + 8 GiB swap |
| GPU | NVIDIA GeForce RTX 3060 |
| CUDA runtime | 13.0 |
| cuDNN | 9.2.0 |
| Python | 3.14.4 |
| PyTorch | 2.13.0 (`+cu130`) |

The GPU/CUDA/runtime values are also recorded automatically in every run's
`metadata.json` file.

### Commands to inspect your own environment

To reproduce on different hardware, you can inspect your environment with:

```bash
# OS and kernel
cat /etc/os-release            # distribution name and version
uname -r                       # kernel version

# CPU
lscpu | grep -E 'Model name|^CPU\(s\):|Socket|Core|Thread'

# Memory
free -h                        # total / available RAM

# Disk (optional)
df -h --total | tail -1

# GPU / CUDA (NVIDIA)
nvidia-smi                     # GPU model + driver + CUDA driver version
nvcc --version                 # CUDA toolkit version (if the toolkit is installed)

# Python and key libraries
python --version
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

## 4. Citing the Puntuguese Corpus

This work builds on the **Puntuguese** corpus. If you use the data, please cite
the original resource:

```bibtex
@inproceedings{inacio-etal-2024-puntuguese,
    title     = "Puntuguese: A Corpus of Puns in {P}ortuguese with Micro-edits",
    author    = "Inacio, Marcio Lima  and
                 Wick-Pedro, Gabriela  and
                 Ramisch, Renata  and
                 Esp{\'i}rito Santo, Lu{\'i}s  and
                 Chacon, Xiomara S. Q.  and
                 Santos, Roney  and
                 Sousa, Rog{\'e}rio  and
                 Anchi{\^e}ta, Rafael  and
                 Goncalo Oliveira, Hugo",
    editor    = "Calzolari, Nicoletta  and
                 Kan, Min-Yen  and
                 Hoste, Veronique  and
                 Lenci, Alessandro  and
                 Sakti, Sakriani  and
                 Xue, Nianwen",
    booktitle = "Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)",
    month     = may,
    year      = "2024",
    address   = "Torino, Italia",
    publisher = "ELRA and ICCL",
    url       = "https://aclanthology.org/2024.lrec-main.1167/",
    pages     = "13332--13343"
}
```

- Corpus: <https://huggingface.co/datasets/Superar/Puntuguese>
- Paper: <https://aclanthology.org/2024.lrec-main.1167/>

---

## 5. Citing This Work

If you use this repository — the reorganized splits, the code, or the results —
please cite the paper associated with it:

```bibtex
@inproceedings{avelar2026split,
    title     = {When the Split Matters: Reorganizing Micro-Edited Pairs for Portuguese Pun Detection},
    author    = {TODO: Author One and Author Two and Author Three},
    booktitle = {Anais do Encontro Nacional de Intelig{\^e}ncia Artificial e Computacional (ENIAC 2026)},
    year      = {2026},
    publisher = {Sociedade Brasileira de Computa{\c c}{\~a}o (SBC)},
    address   = {TODO: City, Brazil},
    note      = {To appear}
    % pages   = {TODO},
    % doi     = {TODO},
    % url     = {TODO}
}
```

> **Fill in before publishing:** replace the `author` list with the final author
> names, and add `address`, `pages`, `doi`, and `url` once the ENIAC 2026
> proceedings are published on the SBC Open Library (SOL). You may also want to
> rename the BibTeX key (`avelar2026split`) to match your group's convention.

---

## 6. Reproducing the Experiments

### Installation

```bash
# Linux/macOS
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

```powershell
# Windows PowerShell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 1. (Re)generate the splits

Run `src/data/make_reorganized_split.ipynb`. It reads the source corpus,
extracts the IDB from each instance, and produces both split strategies
(`pair_controlled` and `max_cross_split`) for every seed under `data/`.

### 2. Train and evaluate the models

Run the notebooks under `src/models/` for each condition:

- `train_bertimbau.ipynb` — BERTimbau baseline;
- `train_ensemble.ipynb` — traditional ensemble.

Each notebook writes its metrics and artifacts under the matching
`results/<model>/<strategy>/seed_<n>/` folder. Aggregated tables
(`summary_mean_std.csv`, `all_runs.csv`) are produced from those per-run files.

### Expected results (mean ± std over 6 seeds)

| Model | `pair_controlled` (clean) | `max_cross_split` (leaky) |
|---|---|---|
| BERTimbau (accuracy) | 0.756 ± 0.010 | 0.691 ± 0.020 |
| Ensemble (accuracy) | 0.471 ± 0.009 | 0.960 ± 0.003 |

BERTimbau is slightly **better** under the clean split, while the ensemble's
apparent strength collapses once micro-edited pairs are prevented from crossing
splits — dropping below the 0.50 majority-class baseline. Small variations may
occur depending on hardware, library versions, and random initialization.

---

## 7. Data Format

The corpus files use the JSONL format. Each line is one independent JSON object:

```json
{ "id": "1.3.H", "text": "Example text", "label": 1 }
```

| Field | Description |
|---|---|
| `id` | Example identifier. The suffix `.H` marks a pun and `.N` its non-pun counterpart; the shared prefix is the Base Identifier (IDB) |
| `text` | Text to be classified |
| `label` | Binary label: `0` for non-pun, `1` for pun |

---

## 8. License

This repository uses a **dual license**, because the data and the code have
different origins:

- **Source code** (everything under `src/`, and the scripts/notebooks) is
  released under the **MIT License** — see [`LICENSE`](LICENSE).
- **Data** (everything under `data/`) is derived from the Puntuguese corpus,
  which is licensed under **CC-BY-SA-4.0**. Because of the *ShareAlike* clause,
  the data in this repository is also distributed under **CC-BY-SA-4.0** and must
  keep the same license and attribution in any derivative — see
  [`LICENSE-DATA`](LICENSE-DATA).

If you redistribute or adapt the data, you must credit both the Puntuguese
authors (Section 4) and this work (Section 5), and keep the CC-BY-SA-4.0 license.
