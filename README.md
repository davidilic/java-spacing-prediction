# Code Formatting Transformer

This project implements a Transformer-based neural network for Java between-token spacing prediction.
It's loosely based on this [paper](https://users.ece.utexas.edu/~gligoric/slides/NieETAL20CoqStyle.pdf) and [presentation](https://users.ece.utexas.edu/~gligoric/slides/NieETAL20CoqStyle.pdf) on formatting Coq code.

## Project Overview

The model learns to predict whitespace and newline placement in source code, using a transformer architecture that considers:
- Token sequences
- Token types
- Scope depth
- Positional information

## Project Structure

```
├── checkpoints/           # Saved model checkpoints
├── dataset/              # Dataset processing and loading
├── evaluation/           # Evaluation notebooks
├── model/                # Model architecture
├── training/             # Training logic
└── requirements.txt      # Project dependencies
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/davidilic/java-spacing-prediction.git
cd java-spacing-prediction
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Creating Dataset (Optional)

This step is optional as the dataset is already provided in the `datasets/data` directory.

The needed Java source files are not included in this repository in order to keep it lightweight. You can use any Java repository you have access to, or download one from GitHub. I used [the android repository](https://github.com/aosp-mirror/platform_frameworks_base).

Process Java source files to create the training dataset:

```bash
python -m dataset.script \
    --repo_path /path/to/java/repo \
    --train_path data/train.jsonl \
    --test_path data/test.jsonl \
    --max_examples 50000 \
    --test_ratio 0.1
```

### Training

Train the model using:

```bash
python main.py
```

This will automatically save models to the `checkpoints` directory.

### Evaluation

Use the provided Jupyter notebook in `evaluation/evaluation.ipynb` to evaluate model performance.

## Pre-trained Model

A pre-trained model checkpoint is available [here](https://drive.google.com/file/d/1Ae-4DFzJ2KZ3vHnP-WFIp1yCDcCsRmoO/view?usp=sharing). To use it:

1. Download the checkpoint file
2. Place it in the `checkpoints` directory
3. Load and evaluate it using the `evaluation/evaluation.ipynb` notebook

## Model Architecture

- Transformer encoder with token, type, and position embeddings
- Dual output heads for:
  - Space count regression
  - Newline classification